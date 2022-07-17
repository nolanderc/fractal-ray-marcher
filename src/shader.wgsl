let ITERATIONS = 256;
let INTERSECTION_THRESHOLD = 1e-4;

struct Uniforms {
    proj_view_inv: mat4x4<f32>,
    near: f32,
    far: f32,
    time: f32,
};

@group(0)
@binding(0)
var<uniform> uniforms: Uniforms;

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
    return max(max(distance.x, distance.y), distance.z);
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

/// Compute the signed distance to the closest object in the scene
fn sdf(ray: vec3<f32>) -> f32 {
    let half = vec3<f32>(0.5, 0.5, 0.5);
    return sub(
        box(ray, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.5)),
        sphere(ray, vec3<f32>(0.0, 0.0, 0.0), map(sin(uniforms.time), -1.0, 1.0, 0.51, 0.8)),
    );
}

fn sdf_normal(ray: vec3<f32>, radius: f32) -> vec3<f32> {
    let dx = vec3<f32>(radius, 0.0, 0.0);
    let dy = vec3<f32>(0.0, radius, 0.0);
    let dz = vec3<f32>(0.0, 0.0, radius);

    let dxdr = (sdf(ray + dx) - sdf(ray - dx)) / 2.0 * radius;
    let dydr = (sdf(ray + dy) - sdf(ray - dy)) / 2.0 * radius;
    let dzdr = (sdf(ray + dz) - sdf(ray - dz)) / 2.0 * radius;

    return normalize(vec3<f32>(dxdr, dydr, dzdr));
}

fn march_ray(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    // how far along the ray we have travelled
    var t = 0.0;

    // the closest we have come to a surface
    var best_distance = 10.0 * uniforms.far;

    // the closest acceptable intersection point
    var best_t = -1.0;

    for (var iteration = 0; iteration < ITERATIONS; iteration++) {
        let ray = t * direction + origin;
        let distance = sdf(ray);

        // do we have an intersection?
        if (distance < t * INTERSECTION_THRESHOLD) {
            if (distance > best_distance) {
                // we have diverged from the best solution
                break;
            }

            // store the current intersection point
            best_distance = distance;
            best_t = t;
        }

        t += distance;
    }

    return best_t;
}

fn background(origin: vec3<f32>, direction: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(abs(direction.xyz), 1.0);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex_main(
    @builtin(vertex_index) index: u32,
) -> VertexOutput {
    let u: f32 = f32(index & 1u);
    let v: f32 = f32(index & 2u);

    let x: f32 = map(u, 0.0, 1.0, -1.0, 1.0);
    let y: f32 = map(v, 0.0, 1.0, -1.0, 1.0);

    var out: VertexOutput;
    out.uv = vec2<f32>(u, v);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn fragment_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let xy = vertex.uv * 2.0 - 1.0;
    var origin_near = uniforms.proj_view_inv * vec4<f32>(xy, 0.0, 1.0) * uniforms.near;
    var origin_far = uniforms.proj_view_inv * vec4<f32>(xy, 1.0, 1.0) * uniforms.far;

    origin_near *= 1.0 / origin_near.w;
    origin_far *= 1.0 / origin_far.w;

    let direction = normalize(origin_far.xyz - origin_near.xyz);
    let origin = origin_near.xyz;

    let distance = march_ray(origin, direction);

    if (distance < 0.0) {
        // no hit
        return background(origin, direction);
    }

    let intersection = distance * direction + origin;

    let normal = sdf_normal(intersection, distance * INTERSECTION_THRESHOLD * 0.5);
    let reflection = reflect(direction, normal);

    return vec4<f32>(normal, 1.0);
}

