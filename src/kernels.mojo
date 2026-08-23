"""Coordinate, time-scale, and celestial WCS kernels exposed through a C ABI."""

from std.math import atan2, cos, floor, sin, sqrt
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime DAYSEC = 86400.0
comptime DEG = 0.017453292519943295769236907684886
comptime T77 = 2443144.5003725
comptime LG = 6.969290134e-10
comptime LB = 1.550519768e-8
comptime TDB0 = -0.0000655
comptime PARALLEL_THRESHOLD = 262144
comptime PARALLEL_CHUNK = 16384


@always_inline
def run_chunks[
    origins: OriginSet, //, func: def(Int) capturing[origins] -> None
](num_work_items: Int):
    """Run chunked work after parallel scheduling moved out of the Mojo SDK."""
    for i in range(num_work_items):
        func(i)


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def hypot2(a: Float64, b: Float64) -> Float64:
    return sqrt(a * a + b * b)


def hypot2_simd[width: Int](
    a: SIMD[DType.float64, width], b: SIMD[DType.float64, width]
) -> SIMD[DType.float64, width]:
    return sqrt(a * a + b * b)


def normalize(day: Float64, frac: Float64, dst1: Ptr, dst2: Ptr, i: Int):
    var carry = floor(frac + 0.5)
    dst1[i] = day + carry
    dst2[i] = frac - carry


def tai_minus_utc(day: Float64) -> Float64:
    if day >= 2457754.5:
        return 37.0
    if day >= 2457204.5:
        return 36.0
    if day >= 2456109.5:
        return 35.0
    if day >= 2454832.5:
        return 34.0
    if day >= 2453736.5:
        return 33.0
    if day >= 2451179.5:
        return 32.0
    if day >= 2450630.5:
        return 31.0
    if day >= 2450083.5:
        return 30.0
    if day >= 2449534.5:
        return 29.0
    if day >= 2449169.5:
        return 28.0
    if day >= 2448804.5:
        return 27.0
    if day >= 2448257.5:
        return 26.0
    if day >= 2447892.5:
        return 25.0
    if day >= 2447161.5:
        return 24.0
    if day >= 2446247.5:
        return 23.0
    if day >= 2445516.5:
        return 22.0
    if day >= 2445151.5:
        return 21.0
    if day >= 2444786.5:
        return 20.0
    if day >= 2444239.5:
        return 19.0
    if day >= 2443874.5:
        return 18.0
    if day >= 2443509.5:
        return 17.0
    if day >= 2443144.5:
        return 16.0
    if day >= 2442778.5:
        return 15.0
    if day >= 2442413.5:
        return 14.0
    if day >= 2442048.5:
        return 13.0
    if day >= 2441683.5:
        return 12.0
    if day >= 2441499.5:
        return 11.0
    return 10.0


def utc_to_tai(a: Float64, b: Float64, dst1: Ptr, dst2: Ptr, i: Int):
    var total = a + b
    var day = floor(total - 0.5) + 0.5
    var frac = (a - day) + b
    var dat = tai_minus_utc(day)
    var leap = tai_minus_utc(day + 1.0) - dat
    normalize(day, frac * (DAYSEC + leap) / DAYSEC + dat / DAYSEC, dst1, dst2, i)


def tai_to_utc(a: Float64, b: Float64, dst1: Ptr, dst2: Ptr, i: Int):
    var total = a + b
    var day = floor(total - 0.5) + 0.5
    var dat = tai_minus_utc(day)
    var leap = tai_minus_utc(day + 1.0) - dat
    var seconds = ((a - day) + b) * DAYSEC - dat
    if seconds < 0.0:
        day -= 1.0
        dat = tai_minus_utc(day)
        leap = tai_minus_utc(day + 1.0) - dat
        seconds = ((a - day) + b) * DAYSEC - dat
    elif seconds >= DAYSEC + leap:
        day += 1.0
        dat = tai_minus_utc(day)
        leap = tai_minus_utc(day + 1.0) - dat
        seconds = ((a - day) + b) * DAYSEC - dat
    normalize(day, seconds / (DAYSEC + leap), dst1, dst2, i)


def tdb_minus_tt(tt1: Float64, tt2: Float64) -> Float64:
    var days = (tt1 - 2451545.0) + tt2
    var t = days / 36525.0
    return (
        0.001657 * sin(628.3076 * t + 6.2401)
        + 0.000022 * sin(575.3385 * t + 4.2970)
        + 0.000014 * sin(1256.6152 * t + 6.1969)
        + 0.000005 * sin(606.9777 * t + 4.0212)
        + 0.000005 * sin(52.9691 * t + 0.4444)
        + 0.000002 * sin(21.3299 * t + 5.5431)
        + 0.000010 * t * sin(628.3076 * t + 4.2490)
    )


def to_tai(a: Float64, b: Float64, scale: Int, tmp1: Ptr, tmp2: Ptr, i: Int):
    if scale == 0:
        utc_to_tai(a, b, tmp1, tmp2, i)
        return
    if scale == 1:
        normalize(a, b, tmp1, tmp2, i)
        return
    var tt1 = a
    var tt2 = b
    if scale == 3:
        var d = ((a - T77) + b) * LG / (1.0 + LG)
        tt2 -= d
    elif scale == 4:
        for _ in range(3):
            tt2 = b - tdb_minus_tt(tt1, tt2) / DAYSEC
    elif scale == 5:
        var tcb_delta = ((a - T77) + b) * LB
        tt2 = b - tcb_delta + TDB0 / DAYSEC
        var tdb2 = tt2
        for _ in range(3):
            tt2 = tdb2 - tdb_minus_tt(tt1, tt2) / DAYSEC
    normalize(tt1, tt2 - 32.184 / DAYSEC, tmp1, tmp2, i)


def from_tai(a: Float64, b: Float64, scale: Int, dst1: Ptr, dst2: Ptr, i: Int):
    if scale == 0:
        tai_to_utc(a, b, dst1, dst2, i)
        return
    if scale == 1:
        normalize(a, b, dst1, dst2, i)
        return
    var tt1 = a
    var tt2 = b + 32.184 / DAYSEC
    if scale == 2:
        normalize(tt1, tt2, dst1, dst2, i)
    elif scale == 3:
        normalize(tt1, tt2 + ((tt1 - T77) + tt2) * LG, dst1, dst2, i)
    elif scale == 4:
        normalize(tt1, tt2 + tdb_minus_tt(tt1, tt2) / DAYSEC, dst1, dst2, i)
    else:
        var tdb2 = tt2 + tdb_minus_tt(tt1, tt2) / DAYSEC
        var delta = (LB * ((tt1 - T77) + tdb2) - TDB0 / DAYSEC) / (1.0 - LB)
        normalize(tt1, tdb2 + delta, dst1, dst2, i)


@export("ma_time_convert")
def ma_time_convert(
    src1_addr: Int,
    src2_addr: Int,
    dst1_addr: Int,
    dst2_addr: Int,
    n: Int,
    src_scale: Int,
    dst_scale: Int,
) abi("C"):
    var src1 = p(src1_addr)
    var src2 = p(src2_addr)
    var dst1 = p(dst1_addr)
    var dst2 = p(dst2_addr)
    for i in range(n):
        to_tai(src1[i], src2[i], src_scale, dst1, dst2, i)
        if dst_scale != 1:
            from_tai(dst1[i], dst2[i], dst_scale, dst1, dst2, i)


def spherical_to_cartesian_range(
    r: Ptr, lat: Ptr, lon: Ptr, x: Ptr, y: Ptr, z: Ptr, begin: Int, end: Int
):
    comptime W = simd_width_of[DType.float64]()
    var i = begin
    while i + W <= end:
        var rv = r.load[width=W](i)
        var latv = lat.load[width=W](i)
        var lonv = lon.load[width=W](i)
        var clat = cos(latv)
        x.store(i, rv * clat * cos(lonv))
        y.store(i, rv * clat * sin(lonv))
        z.store(i, rv * sin(latv))
        i += W
    while i < end:
        var clat = cos(lat[i])
        x[i] = r[i] * clat * cos(lon[i])
        y[i] = r[i] * clat * sin(lon[i])
        z[i] = r[i] * sin(lat[i])
        i += 1


@export("ma_spherical_to_cartesian")
def ma_spherical_to_cartesian(
    r_addr: Int,
    lat_addr: Int,
    lon_addr: Int,
    x_addr: Int,
    y_addr: Int,
    z_addr: Int,
    n: Int,
) abi("C"):
    var r = p(r_addr)
    var lat = p(lat_addr)
    var lon = p(lon_addr)
    var x = p(x_addr)
    var y = p(y_addr)
    var z = p(z_addr)

    @parameter
    def work(chunk: Int):
        var begin = chunk * PARALLEL_CHUNK
        spherical_to_cartesian_range(
            r, lat, lon, x, y, z, begin, min(begin + PARALLEL_CHUNK, n)
        )

    if n >= PARALLEL_THRESHOLD:
        run_chunks[work]((n + PARALLEL_CHUNK - 1) // PARALLEL_CHUNK)
    else:
        spherical_to_cartesian_range(r, lat, lon, x, y, z, 0, n)


@export("ma_cartesian_to_spherical")
def ma_cartesian_to_spherical(
    x_addr: Int,
    y_addr: Int,
    z_addr: Int,
    r_addr: Int,
    lat_addr: Int,
    lon_addr: Int,
    n: Int,
) abi("C"):
    var x = p(x_addr)
    var y = p(y_addr)
    var z = p(z_addr)
    var r = p(r_addr)
    var lat = p(lat_addr)
    var lon = p(lon_addr)
    for i in range(n):
        var h = hypot2(x[i], y[i])
        r[i] = hypot2(h, z[i])
        lat[i] = atan2(z[i], h)
        lon[i] = atan2(y[i], x[i])
        if lon[i] < 0.0:
            lon[i] += 6.283185307179586476925286766559


def angular_separation_range(
    lon1: Ptr, lat1: Ptr, lon2: Ptr, lat2: Ptr, dst: Ptr, begin: Int, end: Int
):
    comptime W = simd_width_of[DType.float64]()
    var i = begin
    while i + W <= end:
        var dl = lon2.load[width=W](i) - lon1.load[width=W](i)
        var lat1v = lat1.load[width=W](i)
        var lat2v = lat2.load[width=W](i)
        var s1 = sin(lat1v)
        var c1 = cos(lat1v)
        var s2 = sin(lat2v)
        var c2 = cos(lat2v)
        var cos_dl = cos(dl)
        var a = c2 * sin(dl)
        var b = c1 * s2 - s1 * c2 * cos_dl
        var c = s1 * s2 + c1 * c2 * cos_dl
        dst.store(i, atan2(hypot2_simd(a, b), c))
        i += W
    while i < end:
        var dl = lon2[i] - lon1[i]
        var s1 = sin(lat1[i])
        var c1 = cos(lat1[i])
        var s2 = sin(lat2[i])
        var c2 = cos(lat2[i])
        var a = c2 * sin(dl)
        var b = c1 * s2 - s1 * c2 * cos(dl)
        var c = s1 * s2 + c1 * c2 * cos(dl)
        dst[i] = atan2(hypot2(a, b), c)
        i += 1


@export("ma_angular_separation")
def ma_angular_separation(
    lon1_addr: Int,
    lat1_addr: Int,
    lon2_addr: Int,
    lat2_addr: Int,
    dst_addr: Int,
    n: Int,
) abi("C"):
    var lon1 = p(lon1_addr)
    var lat1 = p(lat1_addr)
    var lon2 = p(lon2_addr)
    var lat2 = p(lat2_addr)
    var dst = p(dst_addr)

    @parameter
    def work(chunk: Int):
        var begin = chunk * PARALLEL_CHUNK
        angular_separation_range(
            lon1,
            lat1,
            lon2,
            lat2,
            dst,
            begin,
            min(begin + PARALLEL_CHUNK, n),
        )

    if n >= PARALLEL_THRESHOLD:
        run_chunks[work]((n + PARALLEL_CHUNK - 1) // PARALLEL_CHUNK)
    else:
        angular_separation_range(lon1, lat1, lon2, lat2, dst, 0, n)


@export("ma_position_angle")
def ma_position_angle(
    lon1_addr: Int,
    lat1_addr: Int,
    lon2_addr: Int,
    lat2_addr: Int,
    dst_addr: Int,
    n: Int,
) abi("C"):
    var lon1 = p(lon1_addr)
    var lat1 = p(lat1_addr)
    var lon2 = p(lon2_addr)
    var lat2 = p(lat2_addr)
    var dst = p(dst_addr)
    for i in range(n):
        var dl = lon2[i] - lon1[i]
        var a = sin(dl) * cos(lat2[i])
        var b = sin(lat2[i]) * cos(lat1[i]) - cos(lat2[i]) * sin(lat1[i]) * cos(dl)
        dst[i] = atan2(a, b)
        if dst[i] < 0.0:
            dst[i] += 6.283185307179586476925286766559


def rotate_spherical_range(
    lon: Ptr,
    lat: Ptr,
    matrix: Ptr,
    dst_lon: Ptr,
    dst_lat: Ptr,
    begin: Int,
    end: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var i = begin
    while i + W <= end:
        var lonv = lon.load[width=W](i)
        var latv = lat.load[width=W](i)
        var clat = cos(latv)
        var x = clat * cos(lonv)
        var y = clat * sin(lonv)
        var z = sin(latv)
        var rx = matrix[0] * x + matrix[1] * y + matrix[2] * z
        var ry = matrix[3] * x + matrix[4] * y + matrix[5] * z
        var rz = matrix[6] * x + matrix[7] * y + matrix[8] * z
        dst_lon.store(i, atan2(ry, rx))
        dst_lat.store(i, atan2(rz, hypot2_simd(rx, ry)))
        i += W
    while i < end:
        var clat = cos(lat[i])
        var x = clat * cos(lon[i])
        var y = clat * sin(lon[i])
        var z = sin(lat[i])
        var rx = matrix[0] * x + matrix[1] * y + matrix[2] * z
        var ry = matrix[3] * x + matrix[4] * y + matrix[5] * z
        var rz = matrix[6] * x + matrix[7] * y + matrix[8] * z
        dst_lon[i] = atan2(ry, rx)
        dst_lat[i] = atan2(rz, hypot2(rx, ry))
        i += 1


@export("ma_rotate_spherical")
def ma_rotate_spherical(
    lon_addr: Int,
    lat_addr: Int,
    matrix_addr: Int,
    dst_lon_addr: Int,
    dst_lat_addr: Int,
    n: Int,
) abi("C"):
    var lon = p(lon_addr)
    var lat = p(lat_addr)
    var matrix = p(matrix_addr)
    var dst_lon = p(dst_lon_addr)
    var dst_lat = p(dst_lat_addr)

    @parameter
    def work(chunk: Int):
        var begin = chunk * PARALLEL_CHUNK
        rotate_spherical_range(
            lon,
            lat,
            matrix,
            dst_lon,
            dst_lat,
            begin,
            min(begin + PARALLEL_CHUNK, n),
        )

    if n >= PARALLEL_THRESHOLD:
        run_chunks[work]((n + PARALLEL_CHUNK - 1) // PARALLEL_CHUNK)
    else:
        rotate_spherical_range(lon, lat, matrix, dst_lon, dst_lat, 0, n)


def sip_pair[width: Int](
    ac: Ptr,
    bc: Ptr,
    order: Int,
    u: SIMD[DType.float64, width],
    v: SIMD[DType.float64, width],
) -> Tuple[
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
]:
    var stride = order + 1
    var i = order
    var aq = SIMD[DType.float64, width](ac[i * stride])
    var bq = SIMD[DType.float64, width](bc[i * stride])
    var aqv = SIMD[DType.float64, width](0.0)
    var bqv = SIMD[DType.float64, width](0.0)
    var av = aq
    var bv = bq
    var au = SIMD[DType.float64, width](0.0)
    var bu = SIMD[DType.float64, width](0.0)
    var adv = aqv
    var bdv = bqv
    i -= 1
    while i >= 0:
        var max_j = order - i
        aq = SIMD[DType.float64, width](ac[i * stride + max_j])
        bq = SIMD[DType.float64, width](bc[i * stride + max_j])
        aqv = SIMD[DType.float64, width](0.0)
        bqv = SIMD[DType.float64, width](0.0)
        var j = max_j - 1
        while j >= 0:
            aqv = aqv * v + aq
            bqv = bqv * v + bq
            aq = aq * v + ac[i * stride + j]
            bq = bq * v + bc[i * stride + j]
            j -= 1
        au = au * u + av
        bu = bu * u + bv
        av = av * u + aq
        bv = bv * u + bq
        adv = adv * u + aqv
        bdv = bdv * u + bqv
        i -= 1
    return av, bv, au, adv, bu, bdv


def wcs_pix2world_range(
    x: Ptr,
    y: Ptr,
    lon: Ptr,
    lat: Ptr,
    begin: Int,
    end: Int,
    origin: Int,
    crpix1: Float64,
    crpix2: Float64,
    crval1: Float64,
    crval2: Float64,
    cd11: Float64,
    cd12: Float64,
    cd21: Float64,
    cd22: Float64,
    is_tan: Int,
    use_sip: Int,
    ac: Ptr,
    bc: Ptr,
    sip_order: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var ra0 = crval1 * DEG
    var dec0 = crval2 * DEG
    var offset = Float64(1 - origin)
    var i = begin
    while i + W <= end:
        var u = x.load[width=W](i) + offset - crpix1
        var v = y.load[width=W](i) + offset - crpix2
        if use_sip != 0:
            var sip_a, sip_b, _, _, _, _ = sip_pair[W](
                ac, bc, sip_order, u, v
            )
            u += sip_a
            v += sip_b
        var px = cd11 * u + cd12 * v
        var py = cd21 * u + cd22 * v
        if is_tan == 0:
            lon.store(i, crval1 + px)
            lat.store(i, crval2 + py)
        else:
            var xi = px * DEG
            var eta = py * DEG
            var den = cos(dec0) - eta * sin(dec0)
            var ra = ra0 + atan2(xi, den)
            var dec = atan2(
                sin(dec0) + eta * cos(dec0), hypot2_simd(den, xi)
            )
            lon.store(i, ra / DEG)
            lat.store(i, dec / DEG)
            for lane in range(W):
                var index = i + lane
                while lon[index] < 0.0:
                    lon[index] += 360.0
                while lon[index] >= 360.0:
                    lon[index] -= 360.0
        i += W
    while i < end:
        var u = x[i] + Float64(1 - origin) - crpix1
        var v = y[i] + Float64(1 - origin) - crpix2
        if use_sip != 0:
            var sip_a, sip_b, _, _, _, _ = sip_pair[1](
                ac, bc, sip_order, u, v
            )
            u += sip_a
            v += sip_b
        var px = cd11 * u + cd12 * v
        var py = cd21 * u + cd22 * v
        if is_tan == 0:
            lon[i] = crval1 + px
            lat[i] = crval2 + py
        else:
            var xi = px * DEG
            var eta = py * DEG
            var den = cos(dec0) - eta * sin(dec0)
            var ra = ra0 + atan2(xi, den)
            var dec = atan2(sin(dec0) + eta * cos(dec0), hypot2(den, xi))
            var ra_deg = ra / DEG
            while ra_deg < 0.0:
                ra_deg += 360.0
            while ra_deg >= 360.0:
                ra_deg -= 360.0
            lon[i] = ra_deg
            lat[i] = dec / DEG
        i += 1


@export("ma_wcs_pix2world")
def ma_wcs_pix2world(
    x_addr: Int,
    y_addr: Int,
    lon_addr: Int,
    lat_addr: Int,
    n: Int,
    origin: Int,
    crpix1: Float64,
    crpix2: Float64,
    crval1: Float64,
    crval2: Float64,
    cd11: Float64,
    cd12: Float64,
    cd21: Float64,
    cd22: Float64,
    is_tan: Int,
    use_sip: Int,
    a_addr: Int,
    b_addr: Int,
    sip_order: Int,
) abi("C"):
    var x = p(x_addr)
    var y = p(y_addr)
    var lon = p(lon_addr)
    var lat = p(lat_addr)
    var ac = p(a_addr)
    var bc = p(b_addr)

    @parameter
    def work(chunk: Int):
        var begin = chunk * PARALLEL_CHUNK
        wcs_pix2world_range(
            x,
            y,
            lon,
            lat,
            begin,
            min(begin + PARALLEL_CHUNK, n),
            origin,
            crpix1,
            crpix2,
            crval1,
            crval2,
            cd11,
            cd12,
            cd21,
            cd22,
            is_tan,
            use_sip,
            ac,
            bc,
            sip_order,
        )

    if n >= PARALLEL_THRESHOLD:
        run_chunks[work]((n + PARALLEL_CHUNK - 1) // PARALLEL_CHUNK)
    else:
        wcs_pix2world_range(
            x,
            y,
            lon,
            lat,
            0,
            n,
            origin,
            crpix1,
            crpix2,
            crval1,
            crval2,
            cd11,
            cd12,
            cd21,
            cd22,
            is_tan,
            use_sip,
            ac,
            bc,
            sip_order,
        )


def wcs_world2pix_range(
    lon: Ptr,
    lat: Ptr,
    x: Ptr,
    y: Ptr,
    converged: Ptr,
    begin: Int,
    end: Int,
    origin: Int,
    crpix1: Float64,
    crpix2: Float64,
    crval1: Float64,
    crval2: Float64,
    cd11: Float64,
    cd12: Float64,
    cd21: Float64,
    cd22: Float64,
    is_tan: Int,
    use_sip: Int,
    ac: Ptr,
    bc: Ptr,
    sip_order: Int,
    tolerance: Float64,
    maxiter: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var det_cd = cd11 * cd22 - cd12 * cd21
    var ra0 = crval1 * DEG
    var dec0 = crval2 * DEG
    var offset = Float64(1 - origin)
    var i = begin
    while i + W <= end:
        var lonv = lon.load[width=W](i)
        var latv = lat.load[width=W](i)
        var px = lonv - crval1
        var py = latv - crval2
        if is_tan != 0:
            var ra = lonv * DEG
            var dec = latv * DEG
            var dra = ra - ra0
            var sin_dec = sin(dec)
            var cos_dec = cos(dec)
            var cos_dra = cos(dra)
            var den = (
                sin_dec * sin(dec0) + cos_dec * cos(dec0) * cos_dra
            )
            px = cos_dec * sin(dra) / den / DEG
            py = (
                sin_dec * cos(dec0) - cos_dec * sin(dec0) * cos_dra
            ) / den / DEG
        var target_u = (cd22 * px - cd12 * py) / det_cd
        var target_v = (-cd21 * px + cd11 * py) / det_cd
        var u = target_u
        var v = target_v
        var done = SIMD[DType.float64, W](1.0)
        if use_sip != 0:
            done = SIMD[DType.float64, W](0.0)
            for _ in range(maxiter):
                var av, bv, au, adv, bu, bdv = sip_pair[W](
                    ac, bc, sip_order, u, v
                )
                var f1 = u + av - target_u
                var f2 = v + bv - target_v
                var j11 = 1.0 + au
                var j12 = adv
                var j21 = bu
                var j22 = 1.0 + bdv
                var det = j11 * j22 - j12 * j21
                var du = (j22 * f1 - j12 * f2) / det
                var dv = (-j21 * f1 + j11 * f2) / det
                u -= du
                v -= dv
                if (
                    abs(du).le(tolerance) & abs(dv).le(tolerance)
                ).reduce_and():
                    done = SIMD[DType.float64, W](1.0)
                    break
        x.store(i, u + crpix1 - offset)
        y.store(i, v + crpix2 - offset)
        converged.store(i, done)
        i += W
    while i < end:
        var px = lon[i] - crval1
        var py = lat[i] - crval2
        if is_tan != 0:
            var ra = lon[i] * DEG
            var dec = lat[i] * DEG
            var dra = ra - ra0
            var den = sin(dec) * sin(dec0) + cos(dec) * cos(dec0) * cos(dra)
            px = cos(dec) * sin(dra) / den / DEG
            py = (sin(dec) * cos(dec0) - cos(dec) * sin(dec0) * cos(dra)) / den / DEG
        var target_u = (cd22 * px - cd12 * py) / det_cd
        var target_v = (-cd21 * px + cd11 * py) / det_cd
        var u = target_u
        var v = target_v
        var done = 1.0
        if use_sip != 0:
            done = 0.0
            for _ in range(maxiter):
                var av, bv, au, adv, bu, bdv = sip_pair[1](
                    ac, bc, sip_order, u, v
                )
                var f1 = u + av - target_u
                var f2 = v + bv - target_v
                var j11 = 1.0 + au
                var j12 = adv
                var j21 = bu
                var j22 = 1.0 + bdv
                var det = j11 * j22 - j12 * j21
                var du = (j22 * f1 - j12 * f2) / det
                var dv = (-j21 * f1 + j11 * f2) / det
                u -= du
                v -= dv
                if abs(du) <= tolerance and abs(dv) <= tolerance:
                    done = 1.0
                    break
        x[i] = u + crpix1 - Float64(1 - origin)
        y[i] = v + crpix2 - Float64(1 - origin)
        converged[i] = done
        i += 1


@export("ma_wcs_world2pix")
def ma_wcs_world2pix(
    lon_addr: Int,
    lat_addr: Int,
    x_addr: Int,
    y_addr: Int,
    converged_addr: Int,
    n: Int,
    origin: Int,
    crpix1: Float64,
    crpix2: Float64,
    crval1: Float64,
    crval2: Float64,
    cd11: Float64,
    cd12: Float64,
    cd21: Float64,
    cd22: Float64,
    is_tan: Int,
    use_sip: Int,
    a_addr: Int,
    b_addr: Int,
    sip_order: Int,
    tolerance: Float64,
    maxiter: Int,
) abi("C"):
    var lon = p(lon_addr)
    var lat = p(lat_addr)
    var x = p(x_addr)
    var y = p(y_addr)
    var converged = p(converged_addr)
    var ac = p(a_addr)
    var bc = p(b_addr)

    @parameter
    def work(chunk: Int):
        var begin = chunk * PARALLEL_CHUNK
        var end = min(begin + PARALLEL_CHUNK, n)
        wcs_world2pix_range(
            lon,
            lat,
            x,
            y,
            converged,
            begin,
            end,
            origin,
            crpix1,
            crpix2,
            crval1,
            crval2,
            cd11,
            cd12,
            cd21,
            cd22,
            is_tan,
            use_sip,
            ac,
            bc,
            sip_order,
            tolerance,
            maxiter,
        )

    if n >= PARALLEL_THRESHOLD:
        run_chunks[work]((n + PARALLEL_CHUNK - 1) // PARALLEL_CHUNK)
    else:
        wcs_world2pix_range(
            lon,
            lat,
            x,
            y,
            converged,
            0,
            n,
            origin,
            crpix1,
            crpix2,
            crval1,
            crval2,
            cd11,
            cd12,
            cd21,
            cd22,
            is_tan,
            use_sip,
            ac,
            bc,
            sip_order,
            tolerance,
            maxiter,
        )
