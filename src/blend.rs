/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#![allow(non_snake_case)]

use crate::*;


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

// kMultiply_Mode
// B(Cb, Cs) = Cb x Cs
// multiply uses its own version of blendfunc_byte because sa and da are not needed
fn blendfunc_multiply_byte(sc: i32, dc: i32, sa: i32, da: i32) -> u32 {
    clamp_div255round(sc * (255 - da)  + dc * (255 - sa)  + sc * dc)
}

pub fn multiply(src: u32, dst: u32) -> u32 {
    let sa = get_packed_a32(src) as i32;
    let da = get_packed_a32(dst) as i32;
    pack_argb32(srcover_byte(get_packed_a32(src), get_packed_a32(dst)),
                blendfunc_multiply_byte(get_packed_r32(src) as i32, get_packed_r32(dst) as i32, sa, da),
                blendfunc_multiply_byte(get_packed_g32(src) as i32, get_packed_g32(dst) as i32, sa, da),
                blendfunc_multiply_byte(get_packed_b32(src) as i32, get_packed_b32(dst) as i32, sa, da))
}

fn srcover_byte(a: u32, b: u32) -> u32 {
    a + b - muldiv255(a, b)
}

pub fn screen(src: u32, dst: u32) -> u32 {
    pack_argb32(srcover_byte(get_packed_a32(src), get_packed_a32(dst)),
                srcover_byte(get_packed_r32(src), get_packed_r32(dst)),
                srcover_byte(get_packed_g32(src), get_packed_g32(dst)),
                srcover_byte(get_packed_b32(src), get_packed_b32(dst)))
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
                overlay_byte(get_packed_r32(src), get_packed_r32(dst), sa, da),
                overlay_byte(get_packed_g32(src), get_packed_g32(dst), sa, da),
                overlay_byte(get_packed_b32(src), get_packed_b32(dst), sa, da))
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
                darken_byte(get_packed_r32(src), get_packed_r32(dst), sa, da),
                darken_byte(get_packed_g32(src), get_packed_g32(dst), sa, da),
                darken_byte(get_packed_b32(src), get_packed_b32(dst), sa, da))
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
                lighten_byte(get_packed_r32(src), get_packed_r32(dst), sa, da),
                lighten_byte(get_packed_g32(src), get_packed_g32(dst), sa, da),
                lighten_byte(get_packed_b32(src), get_packed_b32(dst), sa, da))
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
                colordodge_byte(get_packed_r32(src) as i32, get_packed_r32(dst) as i32, sa, da),
                colordodge_byte(get_packed_g32(src) as i32, get_packed_g32(dst) as i32, sa, da),
                colordodge_byte(get_packed_b32(src) as i32, get_packed_b32(dst) as i32, sa, da))
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
                colorburn_byte(get_packed_r32(src) as i32, get_packed_r32(dst) as i32, sa, da),
                colorburn_byte(get_packed_g32(src) as i32, get_packed_g32(dst) as i32, sa, da),
                colorburn_byte(get_packed_b32(src) as i32, get_packed_b32(dst) as i32, sa, da))
}

fn hardlight_byte(sc: i32, dc: i32, sa: i32, da: i32) -> u32 {
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
                hardlight_byte(get_packed_r32(src) as i32, get_packed_r32(dst) as i32, sa, da),
                hardlight_byte(get_packed_g32(src) as i32, get_packed_g32(dst) as i32, sa, da),
                hardlight_byte(get_packed_b32(src) as i32, get_packed_b32(dst) as i32, sa, da))
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
                softlight_byte(get_packed_r32(src) as i32, get_packed_r32(dst) as i32, sa, da),
                softlight_byte(get_packed_g32(src) as i32, get_packed_g32(dst) as i32, sa, da),
                softlight_byte(get_packed_b32(src) as i32, get_packed_b32(dst) as i32, sa, da))
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

fn difference_byte(sc: i32, dc: i32, sa: i32, da: i32)  -> u32 {
    let tmp = (sc * da).min(dc * sa);
    return clamp_signed_byte(sc + dc - 2 * div255(tmp as u32) as i32);
}

pub fn difference(src: u32, dst: u32) -> u32 {
    let sa = get_packed_a32(src) as i32;
    let da = get_packed_a32(dst) as i32;
    pack_argb32(srcover_byte(sa as u32, da as u32),
                difference_byte(get_packed_r32(src) as i32, get_packed_r32(dst) as i32, sa, da),
                difference_byte(get_packed_g32(src) as i32, get_packed_g32(dst) as i32, sa, da),
                difference_byte(get_packed_b32(src) as i32, get_packed_b32(dst) as i32, sa, da))
}

fn exclusion_byte(sc: i32, dc: i32, _sa: i32, _da: i32)  -> u32 {
    // this equations is wacky, wait for SVG to confirm it
    //int r = sc * da + dc * sa - 2 * sc * dc + sc * (255 - da) + dc * (255 - sa);

    // The above equation can be simplified as follows
    let r = 255*(sc + dc) - 2 * sc * dc;
    return clamp_div255round(r);
}

pub fn exclusion(src: u32, dst: u32) -> u32 {
    let sa = get_packed_a32(src) as i32;
    let da = get_packed_a32(dst) as i32;
    pack_argb32(srcover_byte(sa as u32, da as u32),
                exclusion_byte(get_packed_r32(src) as i32, get_packed_r32(dst) as i32, sa, da),
                exclusion_byte(get_packed_g32(src) as i32, get_packed_g32(dst) as i32, sa, da),
                exclusion_byte(get_packed_b32(src) as i32, get_packed_b32(dst) as i32, sa, da))
}

// The CSS compositing spec introduces the following formulas:
// (See https://dvcs.w3.org/hg/FXTF/rawfile/tip/compositing/index.html#blendingnonseparable)
// SkComputeLuminance is similar to this formula but it uses the new definition from Rec. 709
// while PDF and CG uses the one from Rec. Rec. 601
// See http://www.glennchan.info/articles/technical/hd-versus-sd-color-space/hd-versus-sd-color-space.htm
fn lum(r: i32, g: i32, b: i32) -> i32
{
    div255((r * 77 + g * 150 + b * 28) as u32) as i32
}

fn mul_div(numer1: i32, numer2: i32, denom: i32) -> i32{
    let tmp = (numer1 as i64 * numer2 as i64) / denom as i64;
    return tmp as i32
}

fn minimum(a: i32, b: i32, c: i32) -> i32 {
    a.min(b).min(c)
}

fn maximum(a: i32, b: i32, c: i32) -> i32 {
    a.max(b).max(c)
}

fn clip_color(r: &mut i32, g: &mut i32, b: &mut i32, a: i32) {
    let L = lum(*r, *g, *b);
    let n = minimum(*r, *g, *b);
    let x = maximum(*r, *g, *b);
    let denom = L - n;
    if (n < 0) && (denom != 0) { // Compute denom and make sure it's non zero
        *r = L + mul_div(*r - L, L, denom);
        *g = L + mul_div(*g - L, L, denom);
        *b = L + mul_div(*b - L, L, denom);
    }

    let denom = x - L;
    if (x > a) && (denom != 0) { // Compute denom and make sure it's non zero
        let numer = a - L;
        *r = L + mul_div(*r - L, numer, denom);
        *g = L + mul_div(*g - L, numer, denom);
        *b = L + mul_div(*b - L, numer, denom);
    }
}

fn sat(r: i32, g: i32, b: i32) -> i32 {
    maximum(r, g, b) - minimum(r, g, b)
}

fn set_saturation_components(cmin: &mut i32, cmind: &mut i32, cmax: &mut i32, s: i32) {
    if *cmax > *cmin {
        *cmind = mul_div(*cmind - *cmin, s, *cmax - *cmin);
        *cmax = s;
    } else {
        *cmax = 0;
        *cmind = 0;
    }

    *cmin = 0;
}

fn set_sat(r: &mut i32, g: &mut i32, b: &mut i32, s: i32) {
    if *r <= *g {
        if *g <= *b {
            set_saturation_components(r, g, b, s);
        } else if *r <= *b {
            set_saturation_components(r, b, g, s);
        } else {
            set_saturation_components(b, r, g, s);
        }
    } else if *r <= *b {
        set_saturation_components(g, r, b, s);
    } else if *g <= *b {
        set_saturation_components(g, b, r, s);
    } else {
        set_saturation_components(b, g, r, s);
    }
}

fn set_lum(r: &mut i32, g: &mut i32, b: &mut i32, a: i32, l: i32) {
    let d = l - lum(*r, *g, *b);
    *r += d;
    *g += d;
    *b += d;

    clip_color(r, g, b, a);
}

// non-separable blend modes are done in non-premultiplied alpha
fn  blendfunc_nonsep_byte(sc: i32, dc: i32, sa: i32, da: i32, blendval: i32) -> u32 {
    clamp_div255round(sc * (255 - da) +  dc * (255 - sa) + blendval)
}

pub fn hue(src: u32, dst: u32) -> u32 {
    let sr = get_packed_r32(src) as i32;
    let sg = get_packed_g32(src) as i32;
    let sb = get_packed_b32(src) as i32;
    let sa = get_packed_a32(src) as i32;

    let dr = get_packed_r32(dst) as i32;
    let dg = get_packed_g32(dst) as i32;
    let db = get_packed_b32(dst) as i32;
    let da = get_packed_a32(dst) as i32;
    let mut Sr; let mut Sg; let mut Sb;

    if sa != 0 && da != 0 {
        Sr = sr * sa;
        Sg = sg * sa;
        Sb = sb * sa;
        set_sat(&mut Sr, &mut Sg, &mut Sb, sat(dr, dg, db) * sa);
        set_lum(&mut Sr, &mut Sg, &mut Sb, sa * da, lum(dr, dg, db) * sa);
    } else {
        Sr = 0;
        Sg = 0;
        Sb = 0;
    }

    let a = srcover_byte(sa as u32, da as u32);
    let r = blendfunc_nonsep_byte(sr, dr, sa, da, Sr);
    let g = blendfunc_nonsep_byte(sg, dg, sa, da, Sg);
    let b = blendfunc_nonsep_byte(sb, db, sa, da, Sb);
    return pack_argb32(a, r, g, b);
}

pub fn saturation(src: u32, dst: u32) -> u32 {
    let sr = get_packed_r32(src) as i32;
    let sg = get_packed_g32(src) as i32;
    let sb = get_packed_b32(src) as i32;
    let sa = get_packed_a32(src) as i32;

    let dr = get_packed_r32(dst) as i32;
    let dg = get_packed_g32(dst) as i32;
    let db = get_packed_b32(dst) as i32;
    let da = get_packed_a32(dst) as i32;
    let mut Dr; let mut Dg; let mut Db;

    if sa != 0 && da != 0 {
        Dr = dr * sa;
        Dg = dg * sa;
        Db = db * sa;
        set_sat(&mut Dr, &mut Dg, &mut Db, sat(sr, sg, sb) * da);
        set_lum(&mut Dr, &mut Dg, &mut Db, sa * da, lum(dr, dg, db) * sa);
    } else {
        Dr = 0;
        Dg = 0;
        Db = 0;
    }

    let a = srcover_byte(sa as u32, da as u32);
    let r = blendfunc_nonsep_byte(sr, dr, sa, da, Dr);
    let g = blendfunc_nonsep_byte(sg, dg, sa, da, Dg);
    let b = blendfunc_nonsep_byte(sb, db, sa, da, Db);
    return pack_argb32(a, r, g, b);
}

pub fn color(src: u32, dst: u32) -> u32 {
    let sr = get_packed_r32(src) as i32;
    let sg = get_packed_g32(src) as i32;
    let sb = get_packed_b32(src) as i32;
    let sa = get_packed_a32(src) as i32;

    let dr = get_packed_r32(dst) as i32;
    let dg = get_packed_g32(dst) as i32;
    let db = get_packed_b32(dst) as i32;
    let da = get_packed_a32(dst) as i32;
    let mut Sr; let mut Sg; let mut Sb;

    if sa != 0 && da != 0 {
        Sr = sr * sa;
        Sg = sg * sa;
        Sb = sb * sa;
        set_lum(&mut Sr, &mut Sg, &mut Sb, sa * da, lum(dr, dg, db) * sa);
    } else {
        Sr = 0;
        Sg = 0;
        Sb = 0;
    }

    let a = srcover_byte(sa as u32, da as u32);
    let r = blendfunc_nonsep_byte(sr, dr, sa, da, Sr);
    let g = blendfunc_nonsep_byte(sg, dg, sa, da, Sg);
    let b = blendfunc_nonsep_byte(sb, db, sa, da, Sb);
    return pack_argb32(a, r, g, b);
}

// B(Cb, Cs) = SetLum(Cb, Lum(Cs))
// Create a color with the luminosity of the source color and the hue and saturation of the backdrop color.
pub fn luminosity(src: u32, dst: u32) -> u32 {
    let sr = get_packed_r32(src) as i32;
    let sg = get_packed_g32(src) as i32;
    let sb = get_packed_b32(src) as i32;
    let sa = get_packed_a32(src) as i32;

    let dr = get_packed_r32(dst) as i32;
    let dg = get_packed_g32(dst) as i32;
    let db = get_packed_b32(dst) as i32;
    let da = get_packed_a32(dst) as i32;
    let mut Dr; let mut Dg; let mut Db;

    if sa != 0 && da != 0 {
        Dr = dr * sa;
        Dg = dg * sa;
        Db = db * sa;
        set_lum(&mut Dr, &mut Dg, &mut Db, sa * da, lum(sr, sg, sb) * da);
    } else {
        Dr = 0;
        Dg = 0;
        Db = 0;
    }

    let a = srcover_byte(sa as u32, da as u32);
    let r = blendfunc_nonsep_byte(sr, dr, sa, da, Dr);
    let g = blendfunc_nonsep_byte(sg, dg, sa, da, Dg);
    let b = blendfunc_nonsep_byte(sb, db, sa, da, Db);
    return pack_argb32(a, r, g, b);
}



