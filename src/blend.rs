/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

use crate::*;

fn pack_argb32(a: u32, r: u32, g: u32, b: u32) -> u32 {
    assert!(r <= a);
    assert!(g <= a);
    assert!(b <= a);

    return (a << A32_SHIFT) | (r << R32_SHIFT) |
        (g << G32_SHIFT) | (b << B32_SHIFT);
}

fn get_packed_a32(packed: u32) -> u32 { ((packed) << (24 - A32_SHIFT)) >> 24 }
fn get_packed_r32(packed: u32) -> u32 { ((packed) << (24 - R32_SHIFT)) >> 24 }
fn get_packed_g32(packed: u32) -> u32 { ((packed) << (24 - G32_SHIFT)) >> 24 }
fn get_packed_b32(packed: u32) -> u32 { ((packed) << (24 - B32_SHIFT)) >> 24 }



pub fn dst(_src: u32, dst: u32) -> u32 {
    dst
}

pub fn src(src: u32, _dst: u32) -> u32 {
    src
}

pub fn clear(_src: u32, _dst: u32) -> u32 {
    0
}

pub fn src_over(src: u32, dst: u32) -> u32 {
    over(src, dst)
}

pub fn dst_over(src: u32, dst: u32) -> u32 {
    over(dst, src)
}

pub fn src_in(src: u32, dst: u32) -> u32 {
    alpha_mul(src, alpha_to_alpha256(packed_alpha(dst)))
}

pub fn dst_in(src: u32, dst: u32) -> u32 {
    alpha_mul(dst, alpha_to_alpha256(packed_alpha(src)))
}

pub fn src_out(src: u32, dst: u32) -> u32 {
    alpha_mul(src, alpha_to_alpha256(255 - packed_alpha(dst)))
}

pub fn dst_out(src: u32, dst: u32) -> u32 {
    alpha_mul(dst, alpha_to_alpha256(255 - packed_alpha(src)))
}

pub fn src_atop(src: u32, dst: u32) -> u32 {
    let sa = packed_alpha(src);
    let da = packed_alpha(dst);
    let isa = 255 - sa;

    return pack_argb32(da,
                       muldiv255(da, get_packed_r32(src)) +
                           muldiv255(isa, get_packed_r32(dst)),
                       muldiv255(da, get_packed_g32(src)) +
                           muldiv255(isa, get_packed_g32(dst)),
                       muldiv255(da, get_packed_b32(src)) +
                           muldiv255(isa, get_packed_b32(dst)));
}

pub fn dst_atop(src: u32, dst: u32) -> u32 {
    let sa = packed_alpha(src);
    let da = packed_alpha(dst);
    let ida = 255 - da;

    return pack_argb32(sa,
                       muldiv255(ida, get_packed_r32(src)) +
                           muldiv255(sa, get_packed_r32(dst)),
                       muldiv255(ida, get_packed_g32(src)) +
                           muldiv255(sa, get_packed_g32(dst)),
                       muldiv255(ida, get_packed_b32(src)) +
                           muldiv255(sa, get_packed_b32(dst)));
}

pub fn xor(src: u32, dst: u32) -> u32 {
    let sa = packed_alpha(src);
    let da = packed_alpha(dst);
    let isa = 255 - da;
    let ida = 255 - da;

    return pack_argb32(sa + da - (muldiv255(sa, da) * 2),
                       muldiv255(ida, get_packed_r32(src)) +
                           muldiv255(isa, get_packed_r32(dst)),
                       muldiv255(ida, get_packed_g32(src)) +
                           muldiv255(isa, get_packed_g32(dst)),
                       muldiv255(ida, get_packed_b32(src)) +
                           muldiv255(isa, get_packed_b32(dst)));
}

fn saturated_add(a: u32, b: u32) -> u32 {
    debug_assert!(a <= 255);
    debug_assert!(b <= 255);
    let sum = a + b;
    if sum > 255 {
        255
    } else {
        sum
    }
}

pub fn add(src: u32, dst: u32) -> u32 {
    pack_argb32(saturated_add(get_packed_a32(src), get_packed_a32(dst)),
                saturated_add(get_packed_r32(src), get_packed_r32(dst)),
                saturated_add(get_packed_g32(src), get_packed_g32(dst)),
                saturated_add(get_packed_b32(src), get_packed_b32(dst)))
}

pub fn multiply(src: u32, dst: u32) -> u32 {
    pack_argb32(muldiv255(get_packed_a32(src), get_packed_a32(dst)),
                muldiv255(get_packed_a32(src), get_packed_a32(dst)),
                muldiv255(get_packed_a32(src), get_packed_a32(dst)),
                muldiv255(get_packed_a32(src), get_packed_a32(dst)))
}

fn srcover_byte(a: u32, b: u32) -> u32 {
    a + b - muldiv255(a, b)
}

pub fn screen(src: u32, dst: u32) -> u32 {
    pack_argb32(srcover_byte(get_packed_a32(src), get_packed_a32(dst)),
                srcover_byte(get_packed_a32(src), get_packed_a32(dst)),
                srcover_byte(get_packed_a32(src), get_packed_a32(dst)),
                srcover_byte(get_packed_a32(src), get_packed_a32(dst)))
}

fn clamp_div255round(prod: i32) -> u32 {
    if prod <= 0 {
        return 0;
    } else if prod >= 255 * 255 {
        return 255;
    } else {
        return div255(prod as u32);
    }
}

fn overlay_byte(sc: u32, dc: u32, sa: u32, da: u32) -> u32 {
    let tmp = sc * (255 - da) + dc * (255 - sa);
    let rc;
    if 2 * dc <= da {
        rc = 2 * sc * dc;
    } else {
        rc = sa * da - 2 * (da - dc) * (sa - sc);
    }
    clamp_div255round((rc + tmp) as i32)
}

pub fn overlay(src: u32, dst: u32) -> u32 {
    let sa = get_packed_a32(src);
    let da = get_packed_a32(dst);
    pack_argb32(srcover_byte(sa, da),
                overlay_byte(get_packed_a32(src), get_packed_a32(dst), sa, da),
                overlay_byte(get_packed_a32(src), get_packed_a32(dst), sa, da),
                overlay_byte(get_packed_a32(src), get_packed_a32(dst), sa, da))
}

fn darken_byte(sc: u32, dc: u32, sa: u32, da: u32) -> u32 {
    let sd = sc * da;
    let ds = dc * sa;
    if sd < ds {
        // srcover
        return sc + dc - div255(ds);
    } else {
        // dstover
        return dc + sc - div255(sd);
    }
}

pub fn darken(src: u32, dst: u32) -> u32 {
    let sa = get_packed_a32(src);
    let da = get_packed_a32(dst);
    pack_argb32(srcover_byte(sa, da),
                darken_byte(get_packed_a32(src), get_packed_a32(dst), sa, da),
                darken_byte(get_packed_a32(src), get_packed_a32(dst), sa, da),
                darken_byte(get_packed_a32(src), get_packed_a32(dst), sa, da))
}

fn lighten_byte(sc: u32, dc: u32, sa: u32, da: u32) -> u32 {
    let sd = sc * da;
    let ds = dc * sa;
    if sd > ds {
        // srcover
        return sc + dc - div255(ds);
    } else {
        // dstover
        return dc + sc - div255(sd);
    }
}

pub fn lighten(src: u32, dst: u32) -> u32 {
    let sa = get_packed_a32(src);
    let da = get_packed_a32(dst);
    pack_argb32(srcover_byte(sa, da),
                lighten_byte(get_packed_a32(src), get_packed_a32(dst), sa, da),
                lighten_byte(get_packed_a32(src), get_packed_a32(dst), sa, da),
                lighten_byte(get_packed_a32(src), get_packed_a32(dst), sa, da))
}

fn colordodge_byte(sc: i32, dc: i32, sa: i32, da: i32) -> u32 {
    let mut diff = sa - sc;
    let rc;
    if 0 == dc {
        return muldiv255(sc as u32 , (255 - da) as u32);
    } else if 0 == diff {
        rc = sa * da + sc * (255 - da) + dc * (255 - sa);
    } else {
        diff = (dc * sa) / diff;
        rc = sa * (if da < diff { da } else { diff }) + sc * (255 - da) + dc * (255 - sa);
    }
    return clamp_div255round(rc);
}

pub fn colordodge(src: u32, dst: u32) -> u32 {
    let sa = get_packed_a32(src) as i32;
    let da = get_packed_a32(dst) as i32;
    pack_argb32(srcover_byte(sa as u32, da as u32),
                colordodge_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da),
                colordodge_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da),
                colordodge_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da))
}

fn colorburn_byte(sc: i32, dc: i32, sa: i32, da: i32) -> u32 {
    let rc;
    if dc == da {
        rc = sa * da + sc * (255 - da) + dc * (255 - sa);
    } else if 0 == sc {
        return muldiv255(dc as u32 , (255 - sa) as u32);
    } else {
        let tmp = (da - dc) * sa / sc;
        rc = sa * (da - (if da < tmp { da } else { tmp } ))
        + sc * (255 - da) + dc * (255 - sa);
    }
    return clamp_div255round(rc);
}

pub fn colorburn(src: u32, dst: u32) -> u32 {
    let sa = get_packed_a32(src) as i32;
    let da = get_packed_a32(dst) as i32;
    pack_argb32(srcover_byte(sa as u32, da as u32),
                colorburn_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da),
                colorburn_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da),
                colorburn_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da))
}

pub fn hardlight_byte(sc: i32, dc: i32, sa: i32, da: i32) -> u32 {
    let rc;
    if 2 * sc <= sa {
        rc = 2 * sc * dc;
    } else {
        rc = sa * da - 2 * (da - dc) * (sa - sc);
    }
    return clamp_div255round(rc + sc * (255 - da) + dc * (255 - sa));
}

pub fn hardlight(src: u32, dst: u32) -> u32 {
    let sa = get_packed_a32(src) as i32;
    let da = get_packed_a32(dst) as i32;
    pack_argb32(srcover_byte(sa as u32, da as u32),
                hardlight_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da),
                hardlight_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da),
                hardlight_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da))
}

/* www.worldserver.com/turk/computergraphics/FixedSqrt.pdf
*/
fn sqrt_bits(x: i32, count: i32) -> i32 {
    debug_assert!(x >= 0 && count > 0 && count <= 30);

    let mut root = 0;
    let mut rem_hi = 0;
    let mut rem_lo = x;

    loop {
        root <<= 1;

        rem_hi = (rem_hi << 2) | (rem_lo >> 30);
        rem_lo <<= 2;

        let test_div = (root << 1) + 1;
        if rem_hi >= test_div {
            rem_hi -= test_div;
            root += 1;
        }
        if -count < 0 {
            break;
        }
    }

    return root;
}

type U8Cpu = u32;
// returns 255 * sqrt(n/255)
fn sqrt_unit_byte(n: U8Cpu) -> U8Cpu {
    return sqrt_bits(n as i32, 15+4) as u32;
}

fn softlight_byte(sc: i32, dc: i32, sa: i32, da: i32) -> u32 {
    let m = if da != 0 { dc * 256 / da } else { 0 };
    let rc;
    if 2 * sc <= sa {
        rc = dc * (sa + ((2 * sc - sa) * (256 - m) >> 8));
    } else if 4 * dc <= da {
        let tmp = (4 * m * (4 * m + 256) * (m - 256) >> 16) + 7 * m;
        rc = dc * sa + (da * (2 * sc - sa) * tmp >> 8);
    } else {
        let tmp = sqrt_unit_byte(m as u32) as i32 - m;
        rc = dc * sa + (da * (2 * sc - sa) * tmp >> 8);
    }
    return clamp_div255round(rc + sc * (255 - da) + dc * (255 - sa));
}

pub fn softlight(src: u32, dst: u32) -> u32 {
    let sa = get_packed_a32(src) as i32;
    let da = get_packed_a32(dst) as i32;
    pack_argb32(srcover_byte(sa as u32, da as u32),
                softlight_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da),
                softlight_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da),
                softlight_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da))
}


fn clamp_signed_byte(n: i32) -> u32 {
    if n < 0 {
        0
    } else if n > 255 {
        255
    } else {
        n as u32
    }
}

fn difference_byte(sc: i32, dc: i32, sa: i32, da: i32)  -> u32{
    let tmp = (sc * da).min(dc * sa);
    return clamp_signed_byte(sc + dc - 2 * div255(tmp as u32) as i32);
}

pub fn difference(src: u32, dst: u32) -> u32 {
    let sa = get_packed_a32(src) as i32;
    let da = get_packed_a32(dst) as i32;
    pack_argb32(srcover_byte(sa as u32, da as u32),
                difference_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da),
                difference_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da),
                difference_byte(get_packed_a32(src) as i32, get_packed_a32(dst) as i32, sa, da))
}

