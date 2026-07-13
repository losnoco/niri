//! CIE colorimetry: gamut conversion matrices from chromaticity coordinates.
//!
//! Computes the linear-light RGB→RGB matrix between two sets of container primaries, used to
//! composite (shader uniform) and scan out (plane color pipeline CTM) content whose container
//! gamut differs from the output blend space. Both consumers use the *same* matrix from here,
//! so composition and direct scanout stay numerically identical.
//!
//! The math is the standard one: primaries + white point → RGB→XYZ via white-point scaling,
//! Bradford chromatic adaptation between differing white points, and the inverse for the
//! target space.

use smithay::wayland::color::management::Chromaticities;

/// A row-major 3x3 matrix.
pub type Mat3 = [f64; 9];

/// The identity matrix.
pub const IDENTITY: Mat3 = [
    1., 0., 0., //
    0., 1., 0., //
    0., 0., 1.,
];

/// The Bradford cone response matrix.
const BRADFORD: Mat3 = [
    0.8951, 0.2664, -0.1614, //
    -0.7502, 1.7135, 0.0367, //
    0.0389, -0.0685, 1.0296,
];

fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [0.; 9];
    for row in 0..3 {
        for col in 0..3 {
            out[row * 3 + col] = (0..3).map(|k| a[row * 3 + k] * b[k * 3 + col]).sum();
        }
    }
    out
}

fn mul_vec(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

/// Inverts a matrix via the adjugate; returns `None` for (numerically) singular input.
fn invert(m: &Mat3) -> Option<Mat3> {
    let cofactor = |r1: usize, c1: usize, r2: usize, c2: usize| {
        m[r1 * 3 + c1] * m[r2 * 3 + c2] - m[r1 * 3 + c2] * m[r2 * 3 + c1]
    };
    let c00 = cofactor(1, 1, 2, 2);
    let c01 = -cofactor(1, 0, 2, 2);
    let c02 = cofactor(1, 0, 2, 1);
    let det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1. / det;
    Some([
        c00 * inv_det,
        -cofactor(0, 1, 2, 2) * inv_det,
        cofactor(0, 1, 1, 2) * inv_det,
        c01 * inv_det,
        cofactor(0, 0, 2, 2) * inv_det,
        -cofactor(0, 0, 1, 2) * inv_det,
        c02 * inv_det,
        -cofactor(0, 0, 2, 1) * inv_det,
        cofactor(0, 0, 1, 1) * inv_det,
    ])
}

/// Converts a protocol chromaticity coordinate (units of 1e-6) to CIE xy.
fn to_xy(coord: (i32, i32)) -> (f64, f64) {
    (f64::from(coord.0) / 1e6, f64::from(coord.1) / 1e6)
}

/// The XYZ tristimulus of a chromaticity at unit luminance (Y = 1).
fn xy_to_xyz(coord: (i32, i32)) -> Option<[f64; 3]> {
    let (x, y) = to_xy(coord);
    if y.abs() < 1e-9 {
        return None;
    }
    Some([x / y, 1., (1. - x - y) / y])
}

/// The linear RGB→XYZ matrix of a set of primaries with its white point.
fn rgb_to_xyz(c: &Chromaticities) -> Option<Mat3> {
    let r = xy_to_xyz(c.red)?;
    let g = xy_to_xyz(c.green)?;
    let b = xy_to_xyz(c.blue)?;
    let w = xy_to_xyz(c.white)?;

    // Columns are the primaries' XYZ, scaled so that RGB(1,1,1) lands on the white point.
    let primaries = [
        r[0], g[0], b[0], //
        r[1], g[1], b[1], //
        r[2], g[2], b[2],
    ];
    let scale = mul_vec(&invert(&primaries)?, w);
    Some([
        r[0] * scale[0],
        g[0] * scale[1],
        b[0] * scale[2],
        r[1] * scale[0],
        g[1] * scale[1],
        b[1] * scale[2],
        r[2] * scale[0],
        g[2] * scale[1],
        b[2] * scale[2],
    ])
}

/// The Bradford chromatic adaptation matrix between two white points (identity if equal).
fn adaptation(src_white: (i32, i32), dst_white: (i32, i32)) -> Option<Mat3> {
    if src_white == dst_white {
        return Some(IDENTITY);
    }
    let src = mul_vec(&BRADFORD, xy_to_xyz(src_white)?);
    let dst = mul_vec(&BRADFORD, xy_to_xyz(dst_white)?);
    if src.iter().any(|v| v.abs() < 1e-9) {
        return None;
    }
    let gain = [
        dst[0] / src[0],
        0.,
        0., //
        0.,
        dst[1] / src[1],
        0., //
        0.,
        0.,
        dst[2] / src[2],
    ];
    Some(mul(&invert(&BRADFORD)?, &mul(&gain, &BRADFORD)))
}

/// The linear-light RGB→RGB matrix converting content in the `src` container primaries into
/// the `dst` primaries, adapting the white point (Bradford) when they differ.
///
/// Returns `None` for degenerate chromaticities (which a well-behaved client never sends);
/// callers should treat that as "no conversion".
pub fn gamut_matrix(src: &Chromaticities, dst: &Chromaticities) -> Option<Mat3> {
    if src == dst {
        return Some(IDENTITY);
    }
    let src_to_xyz = rgb_to_xyz(src)?;
    let xyz_to_dst = invert(&rgb_to_xyz(dst)?)?;
    let adapt = adaptation(src.white, dst.white)?;
    Some(mul(&xyz_to_dst, &mul(&adapt, &src_to_xyz)))
}

#[cfg(test)]
mod tests {
    use smithay::wayland::color::management::Primaries;

    use super::*;

    fn chroma(p: Primaries) -> Chromaticities {
        Chromaticities::from_named(p)
    }

    fn assert_close(a: &Mat3, b: &Mat3, eps: f64) {
        for i in 0..9 {
            assert!((a[i] - b[i]).abs() < eps, "index {i}: {} vs {}", a[i], b[i]);
        }
    }

    #[test]
    fn srgb_to_bt2020_matches_the_shader_constants() {
        // The blend shaders' to_bt2020/to_bt709 constants (BT.2087 / BT.2407 values, 6 dp).
        let to_bt2020 = [
            0.627404, 0.329283, 0.043313, //
            0.069097, 0.919540, 0.011362, //
            0.016391, 0.088013, 0.895595,
        ];
        let to_bt709 = [
            1.660491, -0.587641, -0.072850, //
            -0.124550, 1.132900, -0.008349, //
            -0.018151, -0.100579, 1.118730,
        ];
        let computed = gamut_matrix(&chroma(Primaries::Srgb), &chroma(Primaries::Bt2020)).unwrap();
        assert_close(&computed, &to_bt2020, 1e-4);
        let computed = gamut_matrix(&chroma(Primaries::Bt2020), &chroma(Primaries::Srgb)).unwrap();
        assert_close(&computed, &to_bt709, 1e-4);
    }

    #[test]
    fn white_maps_to_white() {
        // RGB(1,1,1) in the source space must map to RGB(1,1,1) in the target space, even
        // across differing white points (DCI-P3's white is not D65).
        for (src, dst) in [
            (Primaries::Srgb, Primaries::Bt2020),
            (Primaries::DciP3, Primaries::Bt2020),
            (Primaries::DisplayP3, Primaries::Srgb),
            (Primaries::AdobeRgb, Primaries::Bt2020),
        ] {
            let m = gamut_matrix(&chroma(src), &chroma(dst)).unwrap();
            let white = mul_vec(&m, [1., 1., 1.]);
            for (i, v) in white.iter().enumerate() {
                assert!((v - 1.).abs() < 1e-6, "{src:?} -> {dst:?} channel {i}: {v}");
            }
        }
    }

    #[test]
    fn round_trip_is_identity() {
        let forward =
            gamut_matrix(&chroma(Primaries::DisplayP3), &chroma(Primaries::Bt2020)).unwrap();
        let back = gamut_matrix(&chroma(Primaries::Bt2020), &chroma(Primaries::DisplayP3)).unwrap();
        assert_close(&mul(&forward, &back), &IDENTITY, 1e-9);
    }

    #[test]
    fn same_primaries_is_identity() {
        let m = gamut_matrix(&chroma(Primaries::Srgb), &chroma(Primaries::Srgb)).unwrap();
        assert_eq!(m, IDENTITY);
    }
}
