mod blend;

const BILINEAR_INTERPOLATION_BITS: u32 = 4;

const A32_SHIFT: u32 = 24;
const R32_SHIFT: u32 = 16;
const G32_SHIFT: u32 = 8;
const B32_SHIFT: u32 = 0;

pub use blend::*;

type Alpha256 = u32;

#[derive(Clone, Copy)]
pub struct Image<'a> {
    pub width: i32,
    pub height: i32,
    pub data: &'a [u32],
}

/// t is 0..256
pub fn lerp(a: u32, b: u32, t: u32) -> u32 {
    // we can reduce this to two multiplies
    // http://stereopsis.com/doubleblend.html
    let mask = 0xff00ff;
    let brb = ((b & 0xff00ff) * t) >> 8;
    let bag = ((b >> 8) & 0xff00ff) * t;
    let t = 256 - t;
    let arb = ((a & 0xff00ff) * t) >> 8;
    let aag = ((a >> 8) & 0xff00ff) * t;
    let rb = arb + brb;
    let ag = aag + bag;
    return (rb & mask) | (ag & !mask);
}

#[derive(Clone, Copy, Debug)]
pub struct GradientStop {
    pub position: f32,
    pub color: u32,
}

pub struct GradientSource {
    matrix: MatrixFixedPoint,
    lut: [u32; 257],
}

impl GradientSource {
    pub fn radial_gradient_eval(&self, x: u16, y: u16) -> u32 {
        let p = self.matrix.transform(x, y);
        // there's no chance that p will overflow when squared
        // so it's safe to use sqrt
        let px = p.x as f32;
        let py = p.y as f32;
        let mut distance = (px * px + py * py).sqrt() as u32;
        distance >>= 8;
        if distance > 32768 {
            distance = 32786;
        }
        self.lut[(distance >> 7) as usize]
    }

    pub fn linear_gradient_eval(&self, x: u16, y: u16) -> u32 {
        let p = self.matrix.transform(x, y);
        let mut lx = p.x >> 16;
        if lx >= 256 {
            lx = 256;
        }
        if lx < 0 {
            lx = 0;
        }
        self.lut[lx as usize]
    }
}

#[derive(Clone, Debug)]
pub struct Gradient {
    pub stops: Vec<GradientStop>
}

impl Gradient {
    pub fn make_source(&self, matrix: &MatrixFixedPoint, alpha: u32) -> Box<GradientSource> {
        let mut source = Box::new(GradientSource { matrix: (*matrix).clone(), lut: [0; 257] });
        self.build_lut(&mut source.lut, alpha_to_alpha256(alpha));
        source
    }
    fn build_lut(&self, lut: &mut [u32; 257], alpha: Alpha256) {
        let mut stop_idx = 0;
        let mut stop = &self.stops[stop_idx];

        let mut last_color = alpha_mul(stop.color, alpha);
        let mut last_pos = 0;

        let mut next_color = last_color;
        let mut next_pos = (256. * stop.position) as u32;

        let mut i = 0;
        while i <= 256 {
            while next_pos <= i {
                stop_idx += 1;
                last_color = next_color;
                if stop_idx >= self.stops.len() {
                    stop = &self.stops[self.stops.len() - 1];
                    next_pos = 256;
                    next_color = alpha_mul(stop.color, alpha);
                    break;
                } else {
                    stop = &self.stops[stop_idx];
                }
                next_pos = (256. * stop.position) as u32;
                next_color = alpha_mul(stop.color, alpha);
            }
            let inverse = (256 * 256) / (next_pos - last_pos);
            let mut t = 0;
            // XXX we could actually avoid doing any multiplications inside
            // this loop by accumulating (next_color - last_color)*inverse
            while i <= next_pos {
                lut[i as usize] = lerp(last_color, next_color, t >> 8);
                t += inverse;
                i += 1;
            }
            last_pos = next_pos;
        }
    }
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

        return bitmap.data[(y * bitmap.width + x) as usize];
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

        return bitmap.data[(y * bitmap.width + x) as usize];
    }
}


/* Inspired by Filter_32_opaque from Skia */
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
    distxiy = (distx << 4) - distxy; /* distx * (16 - disty) */
    distixy = (disty << 4) - distxy; /* disty * (16 - distx) */

    /* (16 - distx) * (16 - disty) */
    // The intermediate calculation can underflow so we use
    // wrapping arithmetic to let the compiler know that it's ok
    distixiy = (16u32 * 16)
        .wrapping_sub(disty << 4)
        .wrapping_sub(distx << 4)
        .wrapping_add(distxy);
    debug_assert!(distixiy >= 0);

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

/* Inspired by Filter_32_alpha from Skia */
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
    distxiy = (distx << 4) - distxy; /* distx * (16 - disty) */
    distixy = (disty << 4) - distxy; /* disty * (16 - distx) */
    distixiy = 16 * 16 - (disty << 4) - (distx << 4) + distxy; /* (16 - distx) * (16 - disty) */

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
    (x * (1 << FIXED_FRACTION_BITS) as f32) as i32
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

pub struct PointFixedPoint {
    pub x: Fixed,
    pub y: Fixed,
}

#[derive(Clone)]
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
        // when taking integer parameters we can use a regular mulitply instead of a fixed one
        PointFixedPoint {
            x: x * self.xx + self.xy * y + self.x0,
            y: y * self.yy + self.yx * x + self.y0,
        }
    }
}

fn packed_alpha(x: u32) -> u32 {
    x >> A32_SHIFT
}

// this is an approximation of true 'over' that does a division by 256 instead
// of 255. It is the same style of blending that Skia does.
pub fn over(src: u32, dst: u32) -> u32 {
    let a = packed_alpha(src);
    let a = 256 - a;
    let mask = 0xff00ff;
    let rb = ((dst & 0xff00ff) * a) >> 8;
    let ag = ((dst >> 8) & 0xff00ff) * a;
    src + (rb & mask) | (ag & !mask)
}

pub fn alpha_to_alpha256(alpha: u32) -> u32 {
    alpha + 1
}

/** Calculates 256 - (value * alpha256) / 255 in range [0,256],
 *  for [0,255] value and [0,256] alpha256. */
fn alpha_mul_inv256(value: u32, alpha256: u32) -> u32 {
    let prod = 0xFFFF - value * alpha256;
    return (prod + (prod >> 8)) >> 8;
}

/** Calculates (value * alpha256) / 255 in range [0,256],
 *  for [0,255] value and [0,256] alpha256. */
fn alpha_mul_256(value: u32, alpha256: u32) -> u32 {
    let prod = value * alpha256;
    return (prod + (prod >> 8)) >> 8;
}

pub fn muldiv255(a: u32, b: u32) -> u32 {
    let tmp = a * b + 0x128;
    ((tmp + (tmp >> 8)) >> 8)
}

pub fn div255(a: u32) -> u32 {
    let tmp = a + 0x128;
    ((tmp + (tmp >> 8)) >> 8)
}

pub fn alpha_mul(x: u32, a: Alpha256) -> u32 {
    let mask = 0xFF00FF;

    let src_rb = ((x & mask) * a) >> 8;
    let src_ag = ((x >> 8) & mask) * a;

    return (src_rb & mask) | (src_ag & !mask)
}

// This approximates the division by 255 using a division by 256.
// It matches the behaviour of SkBlendARGB32 from Skia in 2017.
// The behaviour of this function was changed in 2016 by Lee Salzman
// in Skia:40254c2c2dc28a34f96294d5a1ad94a99b0be8a6 to keep more of the
// intermediate precision
pub fn over_in(src: u32, dst: u32, alpha: u32) -> u32 {
    let src_alpha = alpha_to_alpha256(alpha);
    let dst_alpha = alpha_mul_inv256(packed_alpha(src), src_alpha);

    let mask = 0xFF00FF;

    let src_rb = (src & mask) * src_alpha;
    let src_ag = ((src >> 8) & mask) * src_alpha;

    let dst_rb = (dst & mask) * dst_alpha;
    let dst_ag = ((dst >> 8) & mask) * dst_alpha;

    // we sum src and dst before reducing to 8 bit to avoid accumulating rounding errors
    return (((src_rb + dst_rb) >> 8) & mask) | ((src_ag + dst_ag) & !mask);
}

// Similar to over_in but includes an additional clip alpha value
pub fn over_in_in(src: u32, dst: u32, mask: u32, clip: u32) -> u32 {
    let src_alpha = alpha_to_alpha256(mask);
    let src_alpha = alpha_mul_256(src_alpha, clip);
    let dst_alpha = alpha_mul_inv256(packed_alpha(src), src_alpha);

    let mask = 0xFF00FF;

    let src_rb = (src & mask) * src_alpha;
    let src_ag = ((src >> 8) & mask) * src_alpha;

    let dst_rb = (dst & mask) * dst_alpha;
    let dst_ag = ((dst >> 8) & mask) * dst_alpha;

    // we sum src and dst before reducing to 8 bit to avoid accumulating rounding errors
    return (((src_rb + dst_rb) >> 8) & mask) | ((src_ag + dst_ag) & !mask);
}

pub fn alpha_lerp(src: u32, dst: u32, mask: u32, clip: u32) -> u32 {
    let alpha = alpha_mul_256(alpha_to_alpha256(mask), clip);
    return lerp(src, dst, alpha);
}
