use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use atomic::Ordering;
use calloop::ping::{make_ping, Ping};
use calloop::timer::{TimeoutAction, Timer};
use calloop::LoopHandle;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{Client, DisplayHandle};
use smithay::wayland::compositor::{
    add_blocker, Blocker, BlockerState, CompositorHandler, SurfaceBarrier,
};

/// Default time limit, after which the transaction completes.
///
/// Serves to avoid hanging when a client fails to respond to a configure promptly.
const TIME_LIMIT: Duration = Duration::from_millis(300);

/// Transaction between Wayland clients.
///
/// How to use it:
/// 1. Create a transaction with [`Transaction::new()`].
/// 2. Clone it as many times as you need.
/// 3. Before adding the transaction as a commit blocker, remember to call
///    [`Transaction::add_notification()`] to receive a notification when the transaction completes.
/// 4. Before adding the transaction as a commit blocker, remember to call
///    [`Transaction::register_deadline_timer()`] to make sure the transaction completes when
///    reaching the deadline.
/// 5. In your surface pre-commit handler, if the transaction corresponding to that commit isn't
///    ready, get a blocker with [`Transaction::blocker()`] and add it to the surface.
#[derive(Debug, Clone)]
pub struct Transaction {
    inner: Arc<Inner>,
    deadline: Rc<RefCell<Deadline>>,
    barrier: Rc<RefCell<Option<SurfaceBarrier>>>,
}

/// Registry of in-flight [`SurfaceBarrier`]s.
///
/// Surfaces committing in response to a transaction's configure are registered into the
/// transaction's barrier ([`Transaction::register_surface()`]). The barrier becomes ready once
/// every registered surface is only waiting on the barrier itself, i.e. the niri transaction
/// completed and all per-surface readiness blockers (dmabuf / syncobj acquire point) cleared.
///
/// The barrier's notifier merely wakes the event loop; [`SurfaceBarriers::release_ready()`] is
/// called at the start of every refresh cycle and releases the ready barriers, applying all
/// their pending surface states atomically within that same cycle.
#[derive(Debug, Clone)]
pub struct SurfaceBarriers {
    ping: Ping,
    barriers: Rc<RefCell<Vec<SurfaceBarrier>>>,
}

impl SurfaceBarriers {
    /// Creates the registry.
    ///
    /// `ping` must wake the main event loop so that a becoming-ready barrier triggers a refresh
    /// cycle promptly.
    pub fn new(ping: Ping) -> Self {
        Self {
            ping,
            barriers: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn create_barrier(&self) -> SurfaceBarrier {
        let ping = self.ping.clone();
        let barrier = SurfaceBarrier::new(move || ping.ping());
        self.barriers.borrow_mut().push(barrier.clone());
        barrier
    }

    /// Releases all barriers that became ready, applying their pending surface states.
    pub fn release_ready<D: CompositorHandler + 'static>(&self, dh: &DisplayHandle, state: &mut D) {
        // Applying a released barrier's states can complete further transactions and make more
        // barriers ready, so loop until a pass finds nothing to release. The borrow must not be
        // held across release(): applying states runs commit handlers which can register new
        // barriers.
        loop {
            let ready: Vec<SurfaceBarrier> = {
                let mut barriers = self.barriers.borrow_mut();
                barriers.retain(|barrier| !barrier.is_released());
                barriers
                    .iter()
                    .filter(|barrier| barrier.is_ready())
                    .cloned()
                    .collect()
            };
            if ready.is_empty() {
                return;
            }
            for barrier in ready {
                trace!("releasing ready surface barrier");
                barrier.release(dh, state);
            }
        }
    }
}

/// Blocker for a [`Transaction`].
#[derive(Debug)]
pub struct TransactionBlocker(Weak<Inner>);

#[derive(Debug)]
enum Deadline {
    NotRegistered(Instant),
    Registered { remove: Ping },
}

#[derive(Debug)]
struct Inner {
    /// Whether the transaction is completed.
    completed: AtomicBool,
    /// Notifications to send out upon completing the transaction.
    notifications: Mutex<Option<(Sender<Client>, Vec<Client>)>>,
}

impl Transaction {
    /// Creates a new transaction.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner::new()),
            deadline: Rc::new(RefCell::new(Deadline::NotRegistered(
                Instant::now() + TIME_LIMIT,
            ))),
            barrier: Rc::new(RefCell::new(None)),
        }
    }

    /// Registers a surface's pending commit into this transaction's surface barrier.
    ///
    /// This adds a [`TransactionBlocker`] gating the commit on the completion of this
    /// transaction, and a barrier blocker delaying the application of the committed state until
    /// every surface registered into the barrier is ready, at which point they all apply
    /// atomically (see [`SurfaceBarriers`]).
    ///
    /// Must be called from the surface's pre-commit hook.
    pub fn register_surface(&self, surface: &WlSurface, barriers: &SurfaceBarriers) {
        let mut barrier = self.barrier.borrow_mut();
        let barrier = barrier.get_or_insert_with(|| barriers.create_barrier());
        barrier.register_surfaces([surface]);
        add_blocker(surface, self.blocker());
    }

    /// Gets a blocker for this transaction.
    pub fn blocker(&self) -> TransactionBlocker {
        trace!(transaction = ?Arc::as_ptr(&self.inner), "generating blocker");
        TransactionBlocker(Arc::downgrade(&self.inner))
    }

    /// Adds a notification for when this transaction completes.
    pub fn add_notification(&self, sender: Sender<Client>, client: Client) {
        if self.is_completed() {
            error!("tried to add notification to a completed transaction");
            return;
        }

        let mut guard = self.inner.notifications.lock().unwrap();
        guard.get_or_insert((sender, Vec::new())).1.push(client);
    }

    /// Registers this transaction's deadline timer on an event loop.
    pub fn register_deadline_timer<T: CompositorHandler + 'static>(
        &self,
        event_loop: &LoopHandle<'static, T>,
        dh: &DisplayHandle,
    ) {
        let mut cell = self.deadline.borrow_mut();
        if let Deadline::NotRegistered(deadline) = *cell {
            let timer = Timer::from_deadline(deadline);
            let inner = Arc::downgrade(&self.inner);
            let barrier = self.barrier.clone();
            let dh = dh.clone();
            let token = event_loop
                .insert_source(timer, move |_, _, state| {
                    let _span = trace_span!("deadline timer", transaction = ?Weak::as_ptr(&inner))
                        .entered();
                    let _ = &state;

                    // FIXME: come up with some way to control the deadline timer from tests.
                    #[cfg(not(test))]
                    if let Some(inner) = inner.upgrade() {
                        trace!("deadline reached, completing transaction");
                        inner.complete();
                    } else {
                        // We should remove the timer automatically. But this callback can still
                        // just happen to run while the ping callback is scheduled, leading to this
                        // branch being legitimately taken.
                        trace!("transaction completed without removing the timer");
                    }

                    // Also force-release the surface barrier: past the deadline we no longer
                    // wait for every participant to become ready.
                    #[cfg(not(test))]
                    if let Some(barrier) = barrier.borrow_mut().take() {
                        trace!("deadline reached, releasing surface barrier");
                        barrier.release(&dh, state);
                    }
                    #[cfg(test)]
                    let _ = (&barrier, &dh);

                    TimeoutAction::Drop
                })
                .unwrap();

            // Add a ping source that will be used to remove the timer automatically.
            let (ping, source) = make_ping().unwrap();
            let loop_handle = event_loop.clone();
            event_loop
                .insert_source(source, move |_, _, _| {
                    loop_handle.remove(token);
                })
                .unwrap();

            *cell = Deadline::Registered { remove: ping };
        }
    }

    /// Returns whether this transaction has already completed.
    pub fn is_completed(&self) -> bool {
        self.inner.is_completed()
    }

    /// Returns whether this is the last instance of this transaction.
    pub fn is_last(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        let _span = trace_span!("drop", transaction = ?Arc::as_ptr(&self.inner)).entered();

        if self.is_last() {
            // If this was the last transaction, complete it.
            trace!("last transaction dropped, completing");
            self.inner.complete();

            // Also remove the timer.
            if let Deadline::Registered { remove } = &*self.deadline.borrow() {
                remove.ping();
            };
        }
    }
}

impl TransactionBlocker {
    pub fn completed() -> Self {
        Self(Weak::new())
    }
}

impl Blocker for TransactionBlocker {
    fn state(&self) -> BlockerState {
        if self.0.upgrade().is_none_or(|x| x.is_completed()) {
            BlockerState::Released
        } else {
            BlockerState::Pending
        }
    }
}

impl Inner {
    fn new() -> Self {
        Self {
            completed: AtomicBool::new(false),
            notifications: Mutex::new(None),
        }
    }

    fn is_completed(&self) -> bool {
        self.completed.load(Ordering::Relaxed)
    }

    fn complete(&self) {
        self.completed.store(true, Ordering::Relaxed);

        let mut guard = self.notifications.lock().unwrap();
        if let Some((sender, clients)) = guard.take() {
            for client in clients {
                if let Err(err) = sender.send(client) {
                    warn!("error sending blocker notification: {err:?}");
                };
            }
        }
    }
}
