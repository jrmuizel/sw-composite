pub mod blend;

const BILINEAR_INTERPOLATION_BITS: u32 = 4;

const A32_SHIFT: u32 = 24;
const R32_SHIFT: u32 = 16;
const G32_SHIFT: u32 = 8;
const B32_SHIFT: u32 = 0;


type Alpha256 = u32;

/// A unpremultiplied color.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Color(u32);

impl Color {
    pub fn new(a: u8, r: u8, g: u8, b: u8) -> Color {
        Color(
            ((a as u32) << A32_SHIFT) |
            ((r as u32) << R32_SHIFT) |
            ((g as u32) << G32_SHIFT) |
            ((b as u32) << B32_SHIFT)
        )
    }

    /// Get the alpha component.
    pub fn a(self) -> u8 {
        (self.0 >> A32_SHIFT & 0xFF) as u8
    }

    /// Get the red component.
    pub fn r(self) -> u8 {
        (self.0 >> R32_SHIFT & 0xFF) as u8
    }

    /// Get the green component.
    pub fn g(self) -> u8 {
        (self.0 >> G32_SHIFT & 0xFF) as u8
    }

    /// Get the blue component.
    pub fn b(self) -> u8 {
        (self.0 >> B32_SHIFT & 0xFF) as u8
    }
}

#[cfg(test)]
#[test]
fn test_color_argb() {
    assert_eq!(Color::new(1, 2, 3, 4).a(), 1);
    assert_eq!(Color::new(1, 2, 3, 4).r(), 2);
    assert_eq!(Color::new(1, 2, 3, 4).g(), 3);
    assert_eq!(Color::new(1, 2, 3, 4).b(), 4);
}

#[derive(Clone, Copy)]
pub struct Image<'a> {
    pub width: i32,
    pub height: i32,
    pub data: &'a [u32],
}

/// t is 0..256
#[inline]
pub fn lerp(a: u32, b: u32, t: u32) -> u32 {
    // this method is from http://stereopsis.com/doubleblend.html
    let mask = 0xff00ff;
    let brb = b & 0xff00ff;
    let bag = (b >> 8) & 0xff00ff;

    let arb = a & 0xff00ff;
    let aag = (a >> 8) & 0xff00ff;

    let drb = brb.wrapping_sub(arb);
    let dag = bag.wrapping_sub(aag);

    let drb = drb.wrapping_mul(t) >> 8;
    let dag = dag.wrapping_mul(t) >> 8;

    let rb = arb + drb;
    let ag = aag + dag;
    (rb & mask) | ((ag << 8) & !mask)
}

#[cfg(test)]
#[test]
fn test_lerp() {
    for i in 0..=256 {
        assert_eq!(lerp(0xffffffff, 0xffffffff, i), 0xffffffff);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GradientStop {
    pub position: f32,
    pub color: Color,
}

pub struct GradientSource {
    matrix: MatrixFixedPoint,
    lut: [u32; 256],
}

pub struct TwoCircleRadialGradientSource {
    matrix: MatrixFixedPoint,
    c1x: f32,
    c1y: f32,
    r1: f32,
    c2x: f32,
    c2y: f32,
    r2: f32,
    lut: [u32; 256],
}

#[derive(Clone, Copy)]
pub enum Spread {
    Pad,
    Reflect,
    Repeat,
}

/// maps `x` to 0..255 according to `spread`
fn apply_spread(x: i32, spread: Spread) -> i32 {
    match spread {
        Spread::Pad => {
            if x > 255 {
                255
            } else if x < 0 {
                0
            } else {
                x
            }
        }
        Spread::Repeat => {
            x & 255
        }
        Spread::Reflect => {
            // a trick from skia to reflect the bits. 256 -> 255
            let sign = (x << 23) >> 31;
            (x ^ sign) & 255
        }
    }
}

impl GradientSource {
    pub fn radial_gradient_eval(&self, x: u16, y: u16, spread: Spread) -> u32 {
        let p = self.matrix.transform(x, y);
        // there's no chance that p will overflow when squared
        // so it's safe to use sqrt
        let px = p.x as f32;
        let py = p.y as f32;
        let mut distance = (px * px + py * py).sqrt() as i32;
        distance >>= 8;

        self.lut[apply_spread(distance, spread) as usize]
    }

    pub fn linear_gradient_eval(&self, x: u16, y: u16, spread: Spread) -> u32 {
        let p = self.matrix.transform(x, y);
        let lx = p.x >> 8;

        self.lut[apply_spread(lx, spread) as usize]
    }
}
// This is called TwoPointConical in Skia
impl TwoCircleRadialGradientSource {
    pub fn eval(&self, x: u16, y: u16, spread: Spread) -> u32 {
        let p = self.matrix.transform(x, y);
        // XXX: this is slow and bad
        // the derivation is from pixman radial_get_scanline_narrow
        // " Mathematically the gradient can be defined as the family of circles
        //
        //    ((1-t)·c₁ + t·(c₂), (1-t)·r₁ + t·r₂)
        //
        // excluding those circles whose radius would be < 0."
        // i.e. anywhere where r < 0 we return 0 (transparent black).
        let px = p.x as f32 / 65536.;
        let py = p.y as f32 / 65536.;
        let cdx = self.c2x - self.c1x;
        let cdy = self.c2y - self.c1y;
        let pdx = px - self.c1x;
        let pdy = py - self.c1y;
        let dr = self.r2 - self.r1;
        let a = cdx*cdx + cdy*cdy - dr*dr;
        let b = pdx*cdx + pdy*cdy + self.r1*dr;
        let c = pdx*pdx + pdy*pdy - self.r1*self.r1;
        let discr = b*b - a*c;

        let t = if a == 0. {
            let t =  1./2. * (c / b);
            if self.r1 * (1. - t) + t * self.r2 < 0. {
                return 0;
            }
            t
        } else {
            if discr < 0. {
                return 0;
            } else {
                let t1 = (b + discr.sqrt())/a;
                let t2 = (b - discr.sqrt())/a;
                if t1 > t2 {
                    t1
                } else {
                    t2
                }
            }
        };

        self.lut[apply_spread((t * 255.) as i32, spread) as usize]
    }
}

#[derive(Clone, Debug)]
pub struct Gradient {
    pub stops: Vec<GradientStop>
}

impl Gradient {
    pub fn make_source(&self, matrix: &MatrixFixedPoint, alpha: u32) -> Box<GradientSource> {
        let mut source = Box::new(GradientSource { matrix: (*matrix).clone(), lut: [0; 256] });
        self.build_lut(&mut source.lut, alpha_to_alpha256(alpha));
        source
    }

    /// evaluate the gradient for a particular
    /// `t` directly
    #[cfg(test)]
    fn eval(&self, t: f32, spread: Spread) -> Color {
        let t = match spread {
            Spread::Pad => {
                if t > 1. {
                    1.
                } else if t < 0. {
                    0.
                } else {
                    t
                }
            }
            Spread::Repeat => {
                t % 1.
            }
            Spread::Reflect => {
                let k = t % 2.;
                if k > 1. {
                    2. - k
                } else {
                    k
                }
            }
        };

        let mut stop_idx = 0;
        let mut above = &self.stops[stop_idx];
        while stop_idx < self.stops.len()-1 && t > above.position {
            stop_idx += 1;
            above = &self.stops[stop_idx]
        }
        let mut below = above;
        if stop_idx > 0 && below.position > t {
            below = &self.stops[stop_idx-1]
        }
        assert!((t < above.position && t > below.position) ||
            above as *const GradientStop == below as *const GradientStop);

        if above as *const GradientStop == below as *const GradientStop {
            above.color
        } else {
            let diff = above.position - below.position;
            let t = (t - below.position) / diff;
            assert!(t <= 1.);
            Color(lerp(below.color.0, above.color.0, (t * 256. + 0.5) as u32))
        }
    }

    pub fn make_two_circle_source(
        &self,
        c1x: f32,
        c1y: f32,
        r1: f32,
        c2x: f32,
        c2y: f32,
        r2: f32,
        matrix: &MatrixFixedPoint,
        alpha: u32,
    ) -> Box<TwoCircleRadialGradientSource> {
        let mut source = Box::new(TwoCircleRadialGradientSource {
            c1x, c1y, r1, c2x, c2y, r2, matrix: matrix.clone(), lut: [0; 256]
        });
        self.build_lut(&mut source.lut, alpha_to_alpha256(alpha));
        source
    }

    fn build_lut(&self, lut: &mut [u32; 256], alpha: Alpha256) {
        let mut stop_idx = 0;
        let mut stop = &self.stops[stop_idx];

        let mut last_color = alpha_mul(stop.color.0, alpha);

        let mut next_color = last_color;
        let mut next_pos = (255. * stop.position) as u32;

        let mut i = 0;

        const FIXED_SHIFT: u32 = 8;
        const FIXED_ONE: u32 = 1 << FIXED_SHIFT;
        const FIXED_HALF: u32 = FIXED_ONE >> 1;

        while i < 255 {
            while next_pos <= i {
                stop_idx += 1;
                last_color = next_color;
                if stop_idx >= self.stops.len() {
                    stop = &self.stops[self.stops.len() - 1];
                    next_pos = 255;
                    next_color = alpha_mul(stop.color.0, alpha);
                    break;
                } else {
                    stop = &self.stops[stop_idx];
                }
                next_pos = (255. * stop.position) as u32;
                next_color = alpha_mul(stop.color.0, alpha);
            }
            let inverse = (FIXED_ONE * 256) / (next_pos - i);
            let mut t = 0;
            // XXX we could actually avoid doing any multiplications inside
            // this loop by accumulating (next_color - last_color)*inverse
            // that's what Skia does
            while i <= next_pos && i < 255 {
                // stops need to be represented in unpremultipled form otherwise we lose information
                // that we need when lerping between colors
                lut[i as usize] = premultiply(lerp(last_color, next_color, (t + FIXED_HALF) >> FIXED_SHIFT));
                t += inverse;

                i += 1;
            }
        }
        // we manually assign the last stop to ensure that it ends up in the last spot even
        // if there's a stop very close to the end. This also avoids a divide-by-zero when
        // calculating inverse
        lut[255] = premultiply(alpha_mul(self.stops[self.stops.len() - 1].color.0, alpha));
    }

}

#[cfg(test)]
#[test]
fn test_gradient_eval() {
    let white = Color(0xffffffff);
    let black = Color(0x00000000);

    let g = Gradient{ stops: vec![GradientStop { position: 0.5, color: white }]};
    assert_eq!(g.eval(0., Spread::Pad), white);
    assert_eq!(g.eval(1., Spread::Pad), white);

    let g = Gradient{ stops: vec![GradientStop { position: 0.5, color: white },
                                  GradientStop { position: 1., color: black }]};
    assert_eq!(g.eval(0., Spread::Pad), white);
    assert_eq!(g.eval(1., Spread::Pad), black);
    //assert_eq!(g.eval(0.75, Spread::Pad), black);

    let g = Gradient{ stops: vec![GradientStop { position: 0.5, color: white },
                                  GradientStop { position: 0.5, color: black }]};
    assert_eq!(g.eval(0., Spread::Pad), white);
    assert_eq!(g.eval(1., Spread::Pad), black);
    assert_eq!(g.eval(0.5, Spread::Pad), white);

    let g = Gradient {
        stops: vec![
            GradientStop {
                position: 0.5,
                color: Color::new(255, 255, 255, 255),
            },
            GradientStop {
                position: 1.0,
                color: Color::new(0, 0, 0, 0),
            },
        ],
    };

    assert_eq!(g.eval(-0.1, Spread::Pad), white);
    assert_eq!(g.eval(1., Spread::Pad), black);

    let mut lut = [0; 256];
    g.build_lut(&mut lut, 256);
    assert_eq!(lut[0], white.0);
    assert_eq!(lut[1], white.0);
    assert_eq!(lut[255], black.0);
}

#[cfg(test)]
#[test]
fn test_gradient_lut() {
    let white = Color(0xffffffff);
    let black = Color(0x00000000);

    let g = Gradient {
        stops: vec![
            GradientStop { position: 0.0, color: white },
            GradientStop { position: 0.999999761, color: white },
            GradientStop { position: 1., color: black },
        ]
    };

    let mut lut = [0; 256];
    g.build_lut(&mut lut, 256);
    assert_eq!(lut[255], black.0);
}

#[cfg(test)]
#[test]
fn test_gradient_pos_range() {
    let white = Color(0xffffffff);
    let black = Color(0x00000000);

    let g = Gradient {
        stops: vec![
            GradientStop { position: -1.0, color: white },
            GradientStop { position: 1.2, color: black },
        ]
    };

    let mut lut = [0; 256];
    g.build_lut(&mut lut, 256);
    // Must not panic.
}

pub trait PixelFetch {
    fn get_pixel(bitmap: &Image,  x: i32,  y: i32) -> u32;
}


pub struct PadFetch;
impl PixelFetch for PadFetch {
    fn get_pixel(bitmap: &Image, mut x: i32, mut y: i32) -> u32 {
        if x < 0 {
            x = 0;
        }
        if x >= bitmap.width {
            x = bitmap.width - 1;
        }

        if y < 0 {
            y = 0;
        }
        if y >= bitmap.height {
            y = bitmap.height - 1;
        }

        bitmap.data[(y * bitmap.width + x) as usize]
    }
}

pub struct RepeatFetch;
impl PixelFetch for RepeatFetch {
    fn get_pixel(bitmap: &Image, mut x: i32, mut y: i32) -> u32 {

        // XXX: This is a very slow approach to repeating.
        // We should instead do the wrapping in the iterator
        x = x % bitmap.width;
        if x < 0 {
            x = x + bitmap.width;
        }

        y = y % bitmap.height;
        if y < 0 {
            y = y + bitmap.height;
        }

        bitmap.data[(y * bitmap.width + x) as usize]
    }
}


// Inspired by Filter_32_opaque from Skia.
fn bilinear_interpolation(
    tl: u32,
    tr: u32,
    bl: u32,
    br: u32,
    mut distx: u32,
    mut disty: u32,
) -> u32 {
    let distxy;
    let distxiy;
    let distixy;
    let distixiy;
    let mut lo;
    let mut hi;

    distx <<= 4 - BILINEAR_INTERPOLATION_BITS;
    disty <<= 4 - BILINEAR_INTERPOLATION_BITS;

    distxy = distx * disty;
    distxiy = (distx << 4) - distxy; // distx * (16 - disty)
    distixy = (disty << 4) - distxy; // disty * (16 - distx)

    // (16 - distx) * (16 - disty)
    // The intermediate calculation can underflow so we use
    // wrapping arithmetic to let the compiler know that it's ok
    distixiy = (16u32 * 16)
        .wrapping_sub(disty << 4)
        .wrapping_sub(distx << 4)
        .wrapping_add(distxy);

    lo = (tl & 0xff00ff) * distixiy;
    hi = ((tl >> 8) & 0xff00ff) * distixiy;

    lo += (tr & 0xff00ff) * distxiy;
    hi += ((tr >> 8) & 0xff00ff) * distxiy;

    lo += (bl & 0xff00ff) * distixy;
    hi += ((bl >> 8) & 0xff00ff) * distixy;

    lo += (br & 0xff00ff) * distxy;
    hi += ((br >> 8) & 0xff00ff) * distxy;

    ((lo >> 8) & 0xff00ff) | (hi & !0xff00ff)
}

// Inspired by Filter_32_alpha from Skia.
fn bilinear_interpolation_alpha(
    tl: u32,
    tr: u32,
    bl: u32,
    br: u32,
    mut distx: u32,
    mut disty: u32,
    alpha: Alpha256
) -> u32 {
    let distxy;
    let distxiy;
    let distixy;
    let distixiy;
    let mut lo;
    let mut hi;

    distx <<= 4 - BILINEAR_INTERPOLATION_BITS;
    disty <<= 4 - BILINEAR_INTERPOLATION_BITS;

    distxy = distx * disty;
    distxiy = (distx << 4) - distxy; // distx * (16 - disty)
    distixy = (disty << 4) - distxy; // disty * (16 - distx)
    // (16 - distx) * (16 - disty)
    // The intermediate calculation can underflow so we use
    // wrapping arithmetic to let the compiler know that it's ok
    distixiy = (16u32 * 16)
        .wrapping_sub(disty << 4)
        .wrapping_sub(distx << 4)
        .wrapping_add(distxy);

    lo = (tl & 0xff00ff) * distixiy;
    hi = ((tl >> 8) & 0xff00ff) * distixiy;

    lo += (tr & 0xff00ff) * distxiy;
    hi += ((tr >> 8) & 0xff00ff) * distxiy;

    lo += (bl & 0xff00ff) * distixy;
    hi += ((bl >> 8) & 0xff00ff) * distixy;

    lo += (br & 0xff00ff) * distxy;
    hi += ((br >> 8) & 0xff00ff) * distxy;

    lo = ((lo >> 8) & 0xff00ff) * alpha;
    hi = ((hi >> 8) & 0xff00ff) * alpha;

    ((lo >> 8) & 0xff00ff) | (hi & !0xff00ff)
}

const FIXED_FRACTION_BITS: u32 = 16;
pub const FIXED_ONE: i32 = 1 << FIXED_FRACTION_BITS;
const FIXED_HALF: i32 = FIXED_ONE >> 1;

fn bilinear_weight(x: Fixed) -> u32 {
    // discard the unneeded bits of precision
    let reduced = x >> (FIXED_FRACTION_BITS - BILINEAR_INTERPOLATION_BITS);
    // extract the remaining fraction
    let fraction = reduced & ((1 << BILINEAR_INTERPOLATION_BITS) - 1);
    fraction as u32
}

type Fixed = i32;

fn fixed_to_int(x: Fixed) -> i32 {
    x >> FIXED_FRACTION_BITS
}

// there are various tricks the can be used
// to make this faster. Let's just do simplest
// thing for now
pub fn float_to_fixed(x: f32) -> Fixed {
    ((x * (1 << FIXED_FRACTION_BITS) as f32) + 0.5) as i32
}

pub fn fetch_bilinear<Fetch: PixelFetch>(image: &Image, x: Fixed, y: Fixed) -> u32 {
    let dist_x = bilinear_weight(x);
    let dist_y = bilinear_weight(y);

    let x1 = fixed_to_int(x);
    let y1 = fixed_to_int(y);
    let x2 = x1 + 1;
    let y2 = y1 + 1;

    let tl = Fetch::get_pixel(image, x1, y1);
    let tr = Fetch::get_pixel(image, x2, y1);
    let bl = Fetch::get_pixel(image, x1, y2);
    let br = Fetch::get_pixel(image, x2, y2);

    bilinear_interpolation(tl, tr, bl, br, dist_x, dist_y)
}

pub fn fetch_bilinear_alpha<Fetch: PixelFetch>(image: &Image, x: Fixed, y: Fixed, alpha: Alpha256) -> u32 {
    let dist_x = bilinear_weight(x);
    let dist_y = bilinear_weight(y);

    let x1 = fixed_to_int(x);
    let y1 = fixed_to_int(y);
    let x2 = x1 + 1;
    let y2 = y1 + 1;

    let tl = Fetch::get_pixel(image, x1, y1);
    let tr = Fetch::get_pixel(image, x2, y1);
    let bl = Fetch::get_pixel(image, x1, y2);
    let br = Fetch::get_pixel(image, x2, y2);

    bilinear_interpolation_alpha(tl, tr, bl, br, dist_x, dist_y, alpha)
}

pub fn fetch_nearest<Fetch: PixelFetch>(image: &Image, x: Fixed, y: Fixed) -> u32 {
    Fetch::get_pixel(image, fixed_to_int(x + FIXED_HALF), fixed_to_int(y + FIXED_HALF))
}

pub fn fetch_nearest_alpha<Fetch: PixelFetch>(image: &Image, x: Fixed, y: Fixed, alpha: Alpha256) -> u32 {
    alpha_mul(Fetch::get_pixel(image, fixed_to_int(x + FIXED_HALF), fixed_to_int(y + FIXED_HALF)), alpha)
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PointFixedPoint {
    pub x: Fixed,
    pub y: Fixed,
}

#[derive(Clone, Debug)]
pub struct MatrixFixedPoint {
    pub xx: Fixed,
    pub xy: Fixed,
    pub yx: Fixed,
    pub yy: Fixed,
    pub x0: Fixed,
    pub y0: Fixed,
}

impl MatrixFixedPoint {
    pub fn transform(&self, x: u16, y: u16) -> PointFixedPoint {
        let x = x as i32;
        let y = y as i32;
        // When taking integer parameters we can use a regular multiply instead of a fixed one.
        //
        // We're also using wrapping to prevent overflow panics in debug.
        // Therefore usage of large numbers is an undefined behavior.
        PointFixedPoint {
            x: x.wrapping_mul(self.xx).wrapping_add(self.xy.wrapping_mul(y)).wrapping_add(self.x0),
            y: y.wrapping_mul(self.yy).wrapping_add(self.yx.wrapping_mul(x)).wrapping_add(self.y0),
        }
    }
}

#[cfg(test)]
#[test]
fn test_large_matrix() {
    let matrix = MatrixFixedPoint {
        xx: std::i32::MAX, xy: std::i32::MAX, yx: std::i32::MAX,
        yy: std::i32::MAX, x0: std::i32::MAX, y0: std::i32::MAX,
    };
    // `transform()` must not panic
    assert_eq!(
        matrix.transform(std::u16::MAX, std::u16::MAX),
        PointFixedPoint { x: 2147352577, y: 2147352577 }
    );
}

fn premultiply(c: u32) -> u32 {
    // This could be optimized by using SWAR
    let a = get_packed_a32(c);
    let mut r = get_packed_r32(c);
    let mut g = get_packed_g32(c);
    let mut b = get_packed_b32(c);

    if a < 255 {
        r = muldiv255(r, a);
        g = muldiv255(g, a);
        b = muldiv255(b, a);
    }

    pack_argb32(a, r, g, b)
}

#[inline]
fn pack_argb32(a: u32, r: u32, g: u32, b: u32) -> u32 {
    debug_assert!(r <= a);
    debug_assert!(g <= a);
    debug_assert!(b <= a);
    (a << A32_SHIFT) | (r << R32_SHIFT) | (g << G32_SHIFT) | (b << B32_SHIFT)
}

#[inline]
fn get_packed_a32(packed: u32) -> u32 { ((packed) << (24 - A32_SHIFT)) >> 24 }
#[inline]
fn get_packed_r32(packed: u32) -> u32 { ((packed) << (24 - R32_SHIFT)) >> 24 }
#[inline]
fn get_packed_g32(packed: u32) -> u32 { ((packed) << (24 - G32_SHIFT)) >> 24 }
#[inline]
fn get_packed_b32(packed: u32) -> u32 { ((packed) << (24 - B32_SHIFT)) >> 24 }

#[inline]
fn packed_alpha(x: u32) -> u32 {
    x >> A32_SHIFT
}

// this is an approximation of true 'over' that does a division by 256 instead
// of 255. It is the same style of blending that Skia does. It corresponds 
// to Skia's SKPMSrcOver
#[inline]
pub fn over(src: u32, dst: u32) -> u32 {
    let a = packed_alpha(src);
    let a = 256 - a;
    let mask = 0xff00ff;
    let rb = ((dst & 0xff00ff) * a) >> 8;
    let ag = ((dst >> 8) & 0xff00ff) * a;
    src + (rb & mask) | (ag & !mask)
}

#[inline]
pub fn alpha_to_alpha256(alpha: u32) -> u32 {
    alpha + 1
}

// Calculates 256 - (value * alpha256) / 255 in range [0,256],
// for [0,255] value and [0,256] alpha256.
#[inline]
fn alpha_mul_inv256(value: u32, alpha256: u32) -> u32 {
    let prod = value * alpha256;
    256 - ((prod + (prod >> 8)) >> 8)
}

// Calculates (value * alpha256) / 255 in range [0,256],
// for [0,255] value and [0,256] alpha256.
fn alpha_mul_256(value: u32, alpha256: u32) -> u32 {
    let prod = value * alpha256;
    (prod + (prod >> 8)) >> 8
}

// Calculates floor(a*b/255 + 0.5)
#[inline]
pub fn muldiv255(a: u32, b: u32) -> u32 {
    // The deriviation for this formula can be
    // found in "Three Wrongs Make a Right" by Jim Blinn.
    let tmp = a * b + 128;
    (tmp + (tmp >> 8)) >> 8
}

// Calculates floor(a/255 + 0.5)
pub fn div255(a: u32) -> u32 {
    // The deriviation for this formula can be
    // found in "Three Wrongs Make a Right" by Jim Blinn.
    let tmp = a + 128;
    (tmp + (tmp >> 8)) >> 8
}

#[inline]
pub fn alpha_mul(x: u32, a: Alpha256) -> u32 {
    let mask = 0xFF00FF;

    let src_rb = ((x & mask) * a) >> 8;
    let src_ag = ((x >> 8) & mask) * a;

    (src_rb & mask) | (src_ag & !mask)
}

// This approximates the division by 255 using a division by 256.
// It matches the behaviour of SkBlendARGB32 from Skia in 2017.
// The behaviour of SkBlendARGB32 was changed in 2016 by Lee Salzman
// in Skia:40254c2c2dc28a34f96294d5a1ad94a99b0be8a6 to keep more of the
// intermediate precision. This was changed to use the alpha setup code
// from the original implementation and additional precision from the reimplementation.
// this combined approach avoids getting incorrect results when `alpha` is 0
// and is slightly faster. However, it suffered from overflow and so
// was switched back to a modified version the previous one that adds 1
// to result.
#[inline]
pub fn over_in(src: u32, dst: u32, alpha: u32) -> u32 {
    let src_alpha = alpha_to_alpha256(alpha);
    let dst_alpha = alpha_mul_inv256(packed_alpha(src), src_alpha);

    let mask = 0xFF00FF;

    let src_rb = (src & mask) * src_alpha;
    let src_ag = ((src >> 8) & mask) * src_alpha;

    let dst_rb = (dst & mask) * dst_alpha;
    let dst_ag = ((dst >> 8) & mask) * dst_alpha;

    // we sum src and dst before reducing to 8 bit to avoid accumulating rounding errors
    (((src_rb + dst_rb) >> 8) & mask) | ((src_ag + dst_ag) & !mask)
}

pub fn over_in_legacy_lerp(src: u32, dst: u32, alpha: u32) -> u32 {
    let src_scale = alpha_to_alpha256(alpha);
    let dst_scale = alpha_to_alpha256(255 - alpha_mul(packed_alpha(src), src_scale));
    alpha_mul(src, src_scale) + alpha_mul(dst, dst_scale)
}

#[cfg(target_arch = "x86")]
use std::arch::x86::{self as x86_intrinsics, __m128i};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{self as x86_intrinsics, __m128i};

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2")))]
pub fn over_in_row(src: &[u32], dst: &mut [u32], alpha: u32) {
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst = over_in(*src, *dst, alpha as u32);
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"))]
pub fn over_in_row(src: &[u32], dst: &mut [u32], alpha: u32) {
    use x86_intrinsics::{
        _mm_loadu_si128,
        _mm_storeu_si128,
    };

    unsafe {
        let mut len = src.len().min(dst.len());
        let mut src_ptr = src.as_ptr() as *const __m128i;
        let mut dst_ptr = dst.as_mut_ptr() as *mut __m128i;

        while len >= 4 {
            _mm_storeu_si128(dst_ptr, over_in_sse2(_mm_loadu_si128(src_ptr), _mm_loadu_si128(dst_ptr), alpha));
            src_ptr = src_ptr.offset(1);
            dst_ptr = dst_ptr.offset(1);
            len -= 4;
        }
        let mut src_ptr = src_ptr as *const u32;
        let mut dst_ptr = dst_ptr as *mut u32;
        while len >= 1 {
            *dst_ptr = over_in(*src_ptr, *dst_ptr, alpha);
            src_ptr = src_ptr.offset(1);
            dst_ptr = dst_ptr.offset(1);
            len -= 1;
        }
    }
}

// derived from Skia's SkBlendARGB32_SSE2
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"))]
fn over_in_sse2(src: __m128i, dst: __m128i, alpha: u32) -> __m128i {
    use x86_intrinsics::{
        _mm_set1_epi16,
        _mm_set1_epi32,
        _mm_mullo_epi16,
        _mm_add_epi16,
        _mm_sub_epi32,
        _mm_add_epi32,
        _mm_srli_epi16,
        _mm_shufflelo_epi16,
        _mm_shufflehi_epi16,
        _mm_srli_epi32,
        _mm_and_si128,
        _mm_andnot_si128,
        _mm_or_si128,
    };

    #[allow(non_snake_case)]
    pub const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
        ((z << 6) | (y << 4) | (x << 2) | w) as i32
    }

    unsafe {
        let alpha = alpha_to_alpha256(alpha);

        let src_scale = _mm_set1_epi16(alpha as i16);
        // SkAlphaMulInv256(SkGetPackedA32(src), src_scale)
        let mut dst_scale = _mm_srli_epi32(src, 24);
        // High words in dst_scale are 0, so it's safe to multiply with 16-bit src_scale.
        dst_scale = _mm_mullo_epi16(dst_scale, src_scale);
        dst_scale = _mm_add_epi32(dst_scale, _mm_srli_epi32(dst_scale, 8));
        dst_scale = _mm_srli_epi32(dst_scale, 8);
        dst_scale = _mm_sub_epi32(_mm_set1_epi32(0x100), dst_scale);

        // Duplicate scales into 2x16-bit pattern per pixel.
        dst_scale = _mm_shufflelo_epi16(dst_scale, _MM_SHUFFLE(2, 2, 0, 0));
        dst_scale = _mm_shufflehi_epi16(dst_scale, _MM_SHUFFLE(2, 2, 0, 0));

        let mask = _mm_set1_epi32(0x00FF00FF);

        // Unpack the 16x8-bit source/destination into 2 8x16-bit splayed halves.
        let mut src_rb = _mm_and_si128(mask, src);
        let mut src_ag = _mm_srli_epi16(src, 8);
        let mut dst_rb = _mm_and_si128(mask, dst);
        let mut dst_ag = _mm_srli_epi16(dst, 8);

        // Scale them.
        src_rb = _mm_mullo_epi16(src_rb, src_scale);
        src_ag = _mm_mullo_epi16(src_ag, src_scale);
        dst_rb = _mm_mullo_epi16(dst_rb, dst_scale);
        dst_ag = _mm_mullo_epi16(dst_ag, dst_scale);

        // Add the scaled source and destination.
        dst_rb = _mm_add_epi16(src_rb, dst_rb);
        dst_ag = _mm_add_epi16(src_ag, dst_ag);

        // Unsplay the halves back together.
        dst_rb = _mm_srli_epi16(dst_rb, 8);
        dst_ag = _mm_andnot_si128(mask, dst_ag);
        _mm_or_si128(dst_rb, dst_ag)
    }
}

// Similar to over_in but includes an additional clip alpha value
#[inline]
pub fn over_in_in(src: u32, dst: u32, mask: u32, clip: u32) -> u32 {
    let src_alpha = alpha_to_alpha256(mask);
    let src_alpha = alpha_to_alpha256(alpha_mul_256(clip, src_alpha));
    let dst_alpha = alpha_mul_inv256(packed_alpha(src), src_alpha);

    let mask = 0xFF00FF;

    let src_rb = (src & mask) * src_alpha;
    let src_ag = ((src >> 8) & mask) * src_alpha;

    let dst_rb = (dst & mask) * dst_alpha;
    let dst_ag = ((dst >> 8) & mask) * dst_alpha;

    // we sum src and dst before reducing to 8 bit to avoid accumulating rounding errors
    (((src_rb + dst_rb) >> 8) & mask) | ((src_ag + dst_ag) & !mask)
}

#[cfg(test)]
#[test]
fn test_over_in() {
    assert_eq!(over_in(0xff00ff00, 0xffff0000, 0xff), 0xff00ff00);
    let mut dst = [0xffff0000, 0, 0, 0];
    over_in_row(&[0xff00ff00, 0, 0, 0], &mut dst, 0xff);
    assert_eq!(dst[0], 0xff00ff00);
    assert_eq!(over_in_in(0xff00ff00, 0xffff0000, 0xff, 0xff), 0xff00ff00);


    // ensure that blending `color` on top of itself
    // doesn't change the color
    for c in 0..=255 {
        for a in 0..=255 {
            let color = 0xff000000 | c;
            assert_eq!(over_in(color, color, a), color);
            let mut dst = [color, 0, 0, 0];
            over_in_row(&[color, 0, 0, 0], &mut dst, a);
            assert_eq!(dst[0], color);
        }
    }

    // make sure we don't overflow
    for s in 0..=255 {
        for a in 0..=255 {
            let result = over_in(s << 24, 0xff000000, a);
            let mut dst = [0xff000000, 0, 0, 0];
            over_in_row(&[s << 24, 0, 0, 0], &mut dst, a);
            assert_eq!(dst[0], result);
        }
    }

    // test that blending by 0 preserves the destination
    assert_eq!(over_in(0xff000000, 0xff0000ff, 0x0), 0xff0000ff);
    assert_eq!(over_in_legacy_lerp(0xff000000, 0xff0000ff, 0x0), 0xff0000ff);

    // tests that blending is broken with the legacy version
    assert_eq!(over_in_legacy_lerp(0xff2e3338, 0xff2e3338, 127), 0xff2e3238);

    let mut dst = [0xff0000ff, 0, 0, 0];
    over_in_row(&[0xff000000, 0, 0, 0], &mut dst, 0);
    assert_eq!(dst[0], 0xff0000ff);
}

pub fn alpha_lerp(src: u32, dst: u32, mask: u32, clip: u32) -> u32 {
    let alpha = alpha_mul_256(alpha_to_alpha256(mask), clip);
    lerp(src, dst, alpha)
}
