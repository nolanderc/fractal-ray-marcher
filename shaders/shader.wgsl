let ITERATIONS = 1024;
let SHADOW_ITERATIONS = 1024;
let SHADOW_RESOLUTION = 1;
let INTERSECTION_THRESHOLD = 1e-5;
let AMBIENT_OCCLUSION_RESOLUTION = 16;

let AA_SAMPLES = 1;

let PI = 3.141592653589793238462643383279502884197169399375105820;

struct Uniforms {
    proj_view_inv: mat4x4<f32>,
    near: f32,
    far: f32,
    time: f32,
    width: f32,
    height: f32,
};

@group(0)
@binding(0)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex_main(
    @builtin(vertex_index) index: u32,
) -> VertexOutput {
    let u: f32 = f32(index & 1u);
    let v: f32 = f32((index >> 1u) & 1u);

    var out: VertexOutput;
    out.uv = vec2<f32>(u, v);
    out.position = vec4<f32>(2.0 * out.uv - 1.0, 0.0, 1.0);

    return out;
}

fn map(x: f32, min_in: f32, max_in: f32, min_out: f32, max_out: f32) -> f32 {
    let domain = max_in - min_in;
    let range = max_out - min_out;
    let normalized = (x - min_in) / domain;
    return min_out + normalized * range;
}

fn map3(x: vec3<f32>, min_in: f32, max_in: f32, min_out: f32, max_out: f32) -> vec3<f32> {
    let domain = max_in - min_in;
    let range = max_out - min_out;
    let normalized = (x - min_in) / domain;
    return min_out + normalized * range;
}

fn sphere(ray: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    return distance(ray, center) - radius;
}

fn box(ray: vec3<f32>, center: vec3<f32>, size: vec3<f32>) -> f32 {
    let distance = abs(ray - center) - size;
    let outside_distance = length(max(distance, vec3<f32>(0.0)));
    let inside_distance = min(0.0, max(max(distance.x, distance.y), distance.z));
    return outside_distance + inside_distance;
}

/// Volume contained by any of the two surfaces
fn add(a: f32, b: f32) -> f32 {
    return min(a, b);
}

/// Only the volume contained within both surfaces
fn intersection(a: f32, b: f32) -> f32 {
    return max(a, b);
}

/// Only surface contained within the first surface, but not the second
fn sub(a: f32, b: f32) -> f32 {
    return intersection(a, -b);
}

let FRACTAL_ITERATIONS = 8;
let FRACTAL_THRESHOLD = 2.0;

fn fractal_power() -> f32 {
    return map(sin(0.05 * uniforms.time), -1.0, 1.0, 1.0, 8.0);
}

fn mandelbulb_iters(ray: vec3<f32>) -> i32 {
    var z = ray;
    var dr = 1.0;
    var r = 0.0;

    let power = fractal_power();

    for (var i = 0; i < FRACTAL_ITERATIONS; i++) {
        r = length(z);
        if (r > FRACTAL_THRESHOLD) { return i; }

        // convert to polar coordinates
        var theta = acos(z.z / r);
        var phi = atan2(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;

        // scale and rotate the point
        let zr = pow(r, power);
        theta = theta * power;
        phi = phi * power;

        // convert back to cartesian coordinates
        z = zr * vec3<f32>(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
        z += ray;
    }

    return FRACTAL_ITERATIONS;
}

fn mandelbulb(ray: vec3<f32>) -> f32 {
    var z = ray;
    var dr = 1.0;
    var r = 0.0;

    let power = fractal_power();

    for (var i = 0; i < FRACTAL_ITERATIONS; i++) {
        r = length(z);
        if (r > FRACTAL_THRESHOLD) { break; }

        // convert to polar coordinates
        var theta = acos(z.z / r);
        var phi = atan2(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;

        // scale and rotate the point
        let zr = pow(r, power);
        theta = theta * power;
        phi = phi * power;

        // convert back to cartesian coordinates
        z = zr * vec3<f32>(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
        z += ray;
    }

    return 0.5 * log(r)*r / dr;
}

fn scene(ray: vec3<f32>) -> f32 {
    let shadowcaster = box(ray, vec3<f32>(0.2, 0.12, 0.4), vec3<f32>(0.2, 0.6, 0.5));
    return add(
        box(ray, vec3<f32>(0.0, -8.5, 0.0), vec3<f32>(8.0)), 
        add(
            shadowcaster,
            sphere(ray, vec3<f32>(0.0, 0.1 * sin(0.8 * uniforms.time), 0.0), 0.5),
        )
    );
}

/// Compute the signed distance to the closest object in the scene
fn sdf(ray: vec3<f32>) -> f32 {
    return scene(ray);
}

fn sdf_normal(ray: vec3<f32>, radius: f32) -> vec3<f32> {
    let dx = vec3<f32>(radius, 0.0, 0.0);
    let dy = vec3<f32>(0.0, radius, 0.0);
    let dz = vec3<f32>(0.0, 0.0, radius);

    let nx = sdf(ray + dx) - sdf(ray - dx);
    let ny = sdf(ray + dy) - sdf(ray - dy);
    let nz = sdf(ray + dz) - sdf(ray - dz);

    return normalize(vec3<f32>(nx, ny, nz) / (2.0 * radius));
}

struct Intersection {
    /// If we missed: a negative number
    /// If we hit: the distance along the ray to the intersection point
    time: f32,
};

fn march_ray(origin: vec3<f32>, direction: vec3<f32>, max_time: f32) -> Intersection {
    // how far along the ray we have travelled
    var time = sdf(origin);

    // the closest acceptable intersection point
    var hit: Intersection;
    hit.time = -1.0;

    var iteration = 0;
    for (; iteration < ITERATIONS && time < max_time; iteration++) {
        let ray = time * direction + origin;
        let distance = sdf(ray);

        // do we have an hit?
        if (distance < time * INTERSECTION_THRESHOLD) {
            // store the current intersection point
            hit.time = time;
            break;
        }

        time += distance;
    }

    // if (iteration == ITERATIONS && hit.time < 0.0) {
    //     hit.time = max_time;
    //     hit.iterations = iteration;
    // }

    return hit;
}

fn background(origin: vec3<f32>, direction: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(vec3(0.09), 1.0);
}

fn material_color(surface: vec3<f32>) -> vec3<f32> {
    return abs(0.5 + 0.5 * sin(20.0 * surface));
}

fn ambient_occlusion(surface: vec3<f32>, normal: vec3<f32>, radius: f32) -> f32 {
    let steps = AMBIENT_OCCLUSION_RESOLUTION;
    let step_size = radius / f32(steps);

    var occlusion = 0.0;
    var t = 1e-3;
    for (var i = 0; i < steps && t <= radius; i++) {
        t += step_size;
        let distance = sdf(surface + t * normal);
        occlusion += pow(abs(t - distance), 2.0) * pow(t, 2.0);
    }

    return clamp(1.0 - occlusion, 0.0, 1.0);
}

fn shadow(origin: vec3<f32>, direction: vec3<f32>, min_time: f32, max_time: f32, k: f32) -> f32 {
    var res = 1.0;
    var iterations = 0;

    var previous_distance = 1e20;
    for (var t = min_time; t < max_time; iterations++) {
        let distance = sdf(origin + direction * t);

        let y = distance*distance/(2.0 * previous_distance);
        let interpolated_distance = sqrt(distance*distance - y*y);

        res = min(res, k*interpolated_distance/max(0.0, t-y));
        previous_distance = distance;
        t = min(max_time, t + distance / f32(SHADOW_RESOLUTION));

        if (res < 1e-3 || iterations >= SHADOW_ITERATIONS) {
            res = 0.0;
            break;
        }
    }
    res = clamp(res, 0.0, 1.0);
    // return res;
    return res*res*(3.0 - 2.0*res);
}

fn shade(surface: vec3<f32>, normal: vec3<f32>, hit: Intersection) -> vec4<f32> {
    let light_rot = 3.9;
    let light_position = 2.0 * vec3<f32>(cos(0.879 * light_rot), 1.0 + sin(light_rot), sin(0.879 * light_rot));
    let light_dir = normalize(light_position - surface);
    let light_distance = length(light_position - surface);

    let light_color = vec3<f32>(1.0, 1.0, 1.0);

    let ambient_brightness = 0.1;
    let ambient_color = vec3<f32>(1.0);
    let ambient = ambient_brightness * ambient_color;

    // let shadow_hit = march_ray(
    //     surface + 2e1 * hit.time * INTERSECTION_THRESHOLD * normal,
    //     light_dir,
    //     uniforms.far,
    // );
    // let shadow = pow(shadow_hit.smallest_angle / (PI / 2.0), 2.0);
    // let shadow = select(0.0, 1.0, shadow_hit.smallest_angle < 0.2);
    let shadow = shadow(surface, light_dir, 1e-2, light_distance, 1e1);

    let brightness = max(0.0, dot(light_dir, normal));
    let diffuse = light_color * brightness * shadow;

    let occlusion = ambient_occlusion(surface, normal, 0.5);

    let color = occlusion * ambient + diffuse;

    return vec4<f32>(vec3(color), 1.0);
}

fn sample_ray(uv: vec2<f32>) -> vec4<f32> {
    let xy = uv * 2.0 - 1.0;
    var origin_near = uniforms.proj_view_inv * vec4<f32>(xy, -1.0, 1.0) * uniforms.near;
    var origin_far = uniforms.proj_view_inv * vec4<f32>(xy, 1.0, 1.0) * uniforms.far;

    origin_near = origin_near / origin_near.w;
    origin_far = origin_far / origin_far.w;

    let direction = normalize(origin_far.xyz - origin_near.xyz);
    let origin = origin_near.xyz;

    let hit = march_ray(origin, direction, uniforms.far);
    if (hit.time < 0.0) {
        // no hit
        return background(origin, direction);
    }

    let intersection = hit.time * direction + origin;

    let normal = sdf_normal(intersection, 3e1 * hit.time * INTERSECTION_THRESHOLD);

    return shade(intersection, normal, hit);
}

@fragment
fn fragment_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let px = 1.0 / f32(AA_SAMPLES) / uniforms.width;
    let py = 1.0 / f32(AA_SAMPLES) / uniforms.height;

    var sum = vec4<f32>(0.0);
    for (var i = 0; i < AA_SAMPLES*AA_SAMPLES; i++) {
        let dx = f32(i % AA_SAMPLES);
        let dy = f32(i / AA_SAMPLES);
        sum += sample_ray(vertex.uv + vec2<f32>(dx * px, dy * py));
    }

    return sum / f32(AA_SAMPLES*AA_SAMPLES);
}

