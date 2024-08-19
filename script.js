const canvas = document.querySelector("#gpuCanvas");
let latticeState;
async function init() {
  const RES = 1;
  const SCALE = 1;
  const GRID_SIZE_X = 1024;
  const GRID_SIZE_Y = 512;
  canvas.width = GRID_SIZE_X * SCALE;
  canvas.height = GRID_SIZE_Y * SCALE;
  const UPDATE_INTERVAL = 0; // Update every 200ms (5 times/sec)
  const WORKGROUP_SIZE = 8; // 8 * 8 * 1 -> 64 per group
  let step = 0; // Track how many simulation steps have been run

  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice();
  if (!device) {
    console.error('need a browser that supports WebGPU');
    return;
  }
  const context = canvas.getContext("webgpu");
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
  });

  // LBM D2Q9 variant
  const GRID_POINTS = GRID_SIZE_X * GRID_SIZE_Y;
  const DATA_PER_POINT = 9 + 1 + 2 + 1 + 1;

  // format: [0,8] directions + [9] rho + [10,11] D2 + [12] curl + [13] barrier
  latticeState = new Float32Array(GRID_POINTS * DATA_PER_POINT);
  // for (let i = 0; i < latticeState.length; i++) latticeState[i] = i / GRID_POINTS / DATA_PER_POINT;

  // Rendering setup

  const vertices = new Float32Array([
    //   X,    Y,
    -RES, -RES, // Triangle 1
    RES, -RES,
    RES, RES,

    -RES, -RES, // Triangle 2
    RES, RES,
    -RES, RES,
  ]);
  const vertexBuffer = device.createBuffer({
    label: "Grid vertices",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, vertices);

  const vertexBufferLayout = {
    arrayStride: 8,
    attributes: [{
      format: "float32x2",
      offset: 0,
      shaderLocation: 0, // Position, see vertex shader
    }],
  };
  
  const gridShaderModule = device.createShaderModule({
    label: "Grid shader",
    code: /* wgsl */ `
      struct VertexInput {
        @location(0) pos: vec2f,
        @builtin(instance_index) instance: u32,
      };
      
      struct VertexOutput {
        @builtin(position) pos: vec4f,
        @location(0) cell: vec2f,
      };
      
      @group(0) @binding(0) var<uniform> grid: vec2f;
      @group(0) @binding(1) var<storage> gridState: array<f32>;

      
      fn getIndex(gridPoint: vec2f) -> u32 {
        return u32(((gridPoint.y % grid.y) * grid.x +
              (gridPoint.x % grid.x)) * ${DATA_PER_POINT}); // 14 per cell
      }
      
      @vertex
      fn vertexMain(input: VertexInput) -> VertexOutput  {
        let i = f32(input.instance);
        let cell = vec2f(i % grid.x, floor(i / grid.x));
        let cellOffset = cell / grid * 2;
        // let state = f32(gridState[input.instance]);
        let gridPos = (input.pos + 1) / grid - 1 + cellOffset; // .pos * state
        
        var output: VertexOutput;
        output.pos = vec4f(gridPos, 0, 1);
        output.cell = cell;
        return output;
      }

      @fragment
      fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
        let cx = (input.cell / grid).x;
        let c = gridState[getIndex(input.cell) + 12];
        return vec4f(c, 1 - 2 * abs(c - 0.5), 1 - c, 1); // rainbow heatmap
        // let r = c;
        // let b = 1 - c;
        // let g = min(r, b);
        // return vec4f(r, g, b, 1); // red-white-blue heatmap
      }
    `
  });

  // Create a uniform buffer that describes the grid.
  const uniformArray = new Float32Array([GRID_SIZE_X, GRID_SIZE_Y]);
  const uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

  let velocity = 0.1;
  let viscosity = 0.05;
  const paramsArray = new Float32Array([step, velocity, viscosity, 0]);
  const paramsBuffer = device.createBuffer({
    label: "Parameters",
    size: paramsArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsArray)

  // Create an array representing the active state of each grid.
  // const gridStateArray = new Uint32Array(GRID_SIZE_X * GRID_SIZE_Y);


  // const rho = new Float32Array(GRID_POINTS);			// macroscopic density
  // const U = new Float32Array(GRID_POINTS * 2);			// macroscopic velocity: D2
  // const curl = new Float32Array(GRID_POINTS);
  // const barrier = new Uint32Array(GRID_POINTS);		// boolean array of barrier locations


  const latticeStateStorage = [
    device.createBuffer({
      label: "Lattice state A",
      size: latticeState.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
      label: "Lattice state B",
      size: latticeState.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),

    device.createBuffer({
      label: "Lattice state Temp",
      size: latticeState.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
  ];

  // 1 array per data type
  {

    // device.createBuffer({
    //   label: "rho A",
    //   size: rho.byteLength,
    //   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    // }),
    // device.createBuffer({
    //   label: "rho B",
    //   size: rho.byteLength,
    //   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    // }),
    // device.createBuffer({
    //   label: "Vel A",
    //   size: U.byteLength,
    //   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    // }),
    // device.createBuffer({
    //   label: "Vel B",
    //   size: U.byteLength,
    //   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    // }),
    // device.createBuffer({
    //   label: "curl A",
    //   size: curl.byteLength,
    //   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    // }),
    // device.createBuffer({
    //   label: "curl B",
    //   size: curl.byteLength,
    //   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    // }),
    // device.createBuffer({
    //   label: "barrier A",
    //   size: barrier.byteLength,
    //   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    // }),
    // device.createBuffer({
    //   label: "barrier B",
    //   size: barrier.byteLength,
    //   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    // })
  }

  const ZERO_WEIGHT = 4 / 9;
  const CARD_WEIGHT = 1 / 9;
  const DIAG_WEIGHT = 1 / 36;
  const WEIGHTS_ARR = new Float32Array([
    ZERO_WEIGHT,
    CARD_WEIGHT,
    CARD_WEIGHT,
    CARD_WEIGHT,
    CARD_WEIGHT,
    DIAG_WEIGHT,
    DIAG_WEIGHT,
    DIAG_WEIGHT,
    DIAG_WEIGHT
  ])

  // init array
  // for (let i = 0; i < gridStateArray.length; ++i) {
  //   gridStateArray[i] = Math.random() > 0.5 ? 1 : 0;
  // }

  device.queue.writeBuffer(latticeStateStorage[0], 0, latticeState);

  const bindGroupLayout = device.createBindGroupLayout({
    label: "Grid Bind Group Layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        buffer: {} // Grid uniform buffer
      }, {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } // Grid state input buffer
      }, {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" } // Grid state output buffer
      }, {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" } // Grid state temporary buffer
      }, {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {}
      }
    ]
  });

  const bindGroups = [
    device.createBindGroup({
      label: "Grid renderer bind group A",
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: uniformBuffer }
        }, {
          binding: 1,
          resource: { buffer: latticeStateStorage[0] }
        }, {
          binding: 2,
          resource: { buffer: latticeStateStorage[1] }
        }, {
          binding: 3,
          resource: { buffer: latticeStateStorage[2] }
        }, {
          binding: 4,
          resource: { buffer: paramsBuffer }
        }
      ],
    }),
    device.createBindGroup({
      label: "Grid renderer bind group B",
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: uniformBuffer }
        }, {
          binding: 1,
          resource: { buffer: latticeStateStorage[1] }
        }, {
          binding: 2,
          resource: { buffer: latticeStateStorage[0] }
        }, {
          binding: 3,
          resource: { buffer: latticeStateStorage[2] }
        }, {
          binding: 4,
          resource: { buffer: paramsBuffer }
        }
      ],
    })
  ];

  const pipelineLayout = device.createPipelineLayout({
    label: "Grid Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });

  const renderPipeline = device.createRenderPipeline({
    label: "Grid pipeline",
    layout: pipelineLayout,
    vertex: {
      module: gridShaderModule,
      entryPoint: "vertexMain",
      buffers: [vertexBufferLayout]
    },
    fragment: {
      module: gridShaderModule,
      entryPoint: "fragmentMain",
      targets: [{
        format: canvasFormat
      }]
    }
  });

  // Compute setup

  const simulationShaderModule = device.createShaderModule({
    label: "D2Q9 LBM Compute Shader",
    code: /* wgsl */ `
    struct args {
      time: f32,
      velocity: f32,
      viscosity: f32,
      padding: f32
    }
    @group(0) @binding(0) var<uniform> grid: vec2f;
    
    // format: [0,8] directions + [9] rho + [10,11] D2 + [12] curl + [13] barrier
    @group(0) @binding(1) var<storage> gridIn: array<f32>;
    @group(0) @binding(2) var<storage, read_write> gridOut: array<f32>;
    @group(0) @binding(3) var<storage, read_write> gridTemp: array<f32>;
    @group(0) @binding(4) var<uniform> params: args;

    const i_0 = 0;
    const i_n = 1;
    const i_s = 2;
    const i_e = 3;
    const i_w = 4;
    const i_ne = 5;
    const i_se = 6;
    const i_nw = 7;
    const i_sw = 8;
    const i_rho = 9;
    const i_ux = 10;
    const i_uy = 11;
    const i_curl = 12;
    const i_barrier = 13;

    const viscosity = 0.05;
    const velocity = 0.1;

    fn getIndex(gridPoint: vec2u) -> u32 {
      return ((gridPoint.y % u32(grid.y)) * u32(grid.x) +
             (gridPoint.x % u32(grid.x))) * ${DATA_PER_POINT}; // 14 per cell
    }
    fn getIndex2(x: u32, y: u32) -> u32 {
      return ((y % u32(y)) * u32(x) +
             (x % u32(x))) * ${DATA_PER_POINT}; // 14 per cell
    }
    
    // fn gridActive(x: u32, y: u32) -> u32 {
    //   return gridIn[gridIndex(vec2(x, y))];
    // }

    @compute
    @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}) // New line
    fn computeMain(@builtin(global_invocation_id) gridPoint: vec3u) {
      // let time = args().time;
      let i = getIndex2(gridPoint.x, gridPoint.y);
      setEquilibrium(i, vec2f(velocity, 0), 1);
      // execute a timestep
      // set boundaries
      // if (
      //   gridPoint.x == 0 || gridPoint.x == ${GRID_SIZE_X - 1} ||
      //   gridPoint.y == 0 || gridPoint.y == ${GRID_SIZE_Y - 1} ||
      //   time == 0
      // ) {
      //   setEquilibrium(i, vec2f(velocity, 0), 1);
      // }
      collide(i);
      stream(gridPoint.x, gridPoint.y);
    }

    fn setEquilibrium(i: u32, vel: vec2f, r: f32) {
      var rho = r;
      if (r == -1f) {
        rho = gridIn[i + i_rho];
      }
      var cardrho: f32 = 1f / 9f * rho;
      var diagrho: f32 = 1f / 36f * rho;
      var ux3: f32 = 3 * vel.x;
      var uy3: f32 = 3 * vel.y;
      var ux2: f32 = vel.x * vel.x;
      var uy2: f32 = vel.y * vel.y;
      var uxuy2: f32 = 2 * vel.x * vel.y;
      var u2: f32 = ux2 + uy2;
      var u215: f32 = 1.5 * u2;
			gridOut[i + i_0] = 4f/9f * rho * (1 - u215);
			gridOut[i + i_e] = cardrho * (1 + ux3 + 4.5 * ux2 - u215);
			gridOut[i + i_w] = cardrho * (1 - ux3 + 4.5 * ux2 - u215);
			gridOut[i + i_n] = cardrho * (1 + uy3 + 4.5 * uy2 - u215);
			gridOut[i + i_s] = cardrho * (1 - uy3 + 4.5 * uy2 - u215);
			gridOut[i + i_ne] = diagrho * (1 + ux3 + uy3 + 4.5 * (u2 + uxuy2) - u215);
			gridOut[i + i_se] = diagrho * (1 + ux3 - uy3 + 4.5 * (u2 - uxuy2) - u215);
			gridOut[i + i_nw] = diagrho * (1 - ux3 + uy3 + 4.5 * (u2 - uxuy2) - u215);
			gridOut[i + i_sw] = diagrho * (1 - ux3 - uy3 + 4.5 * (u2 + uxuy2) - u215);
			gridOut[i + i_rho] = rho;
			gridOut[i + i_ux] = vel.x;
			gridOut[i + i_uy] = vel.y;
    }

    // move toward equilibrium
    fn collide(i: u32) {
      // compute rho, ux, uy
			var omega: f32 = 1 / (3 * viscosity + 0.5);		// reciprocal of relaxation time

      var rho: f32 = gridIn[i] + gridIn[i + i_n] + gridIn[i + i_s] +
                      gridIn[i + i_e] + gridIn[i + i_w] + gridIn[i + i_ne] +
                      gridIn[i + i_se] + gridIn[i + i_nw] + gridIn[i + i_sw];
      gridOut[i + i_rho] = rho;

      var u_x: f32 = gridIn[i + i_e] - gridIn[i + i_w] + gridIn[i + i_ne] +
                      gridIn[i + i_se] - gridIn[i + i_nw] - gridIn[i + i_sw];
      u_x /= rho;
      gridOut[i + i_ux] = u_x;

      var u_y: f32 = gridIn[i + i_n] - gridIn[i + i_s] + gridIn[i + i_ne] -
                      gridIn[i + i_se] + gridIn[i + i_nw] - gridIn[i + i_sw];
      u_y /= rho;
      gridOut[i + i_uy] = u_y;

      // precompute stuff
      var cardrho: f32 = 1f / 9f * rho;
      var diagrho: f32 = 1f / 36f * rho;
      var ux3: f32 = 3 * u_x;
      var uy3: f32 = 3 * u_y;
      var ux2: f32 = u_x * u_x;
      var uy2: f32 = u_y * u_y;
      var uxuy2: f32 = 2 * u_x * u_y;
      var u2: f32 = ux2 + uy2;
      var u215: f32 = 1.5 * u2;
      // compute equilibrium densities and update
      gridTemp[i + i_0] = gridIn[i + i_0] + omega * (4f / 9f * rho * (1 - u215) - gridIn[i + i_0]);
      gridTemp[i + i_n] = gridIn[i + i_n] + omega * (cardrho * (1 + uy3 + 4.5 * uy2 - u215) - gridIn[i + i_n]);
      gridTemp[i + i_s] = gridIn[i + i_s] + omega * (cardrho * (1 - uy3 + 4.5 * uy2 - u215) - gridIn[i + i_s]);
      gridTemp[i + i_e] = gridIn[i + i_e] + omega * (cardrho * (1 + ux3 + 4.5 * ux2 - u215) - gridIn[i + i_e]);
      gridTemp[i + i_w] = gridIn[i + i_w] + omega * (cardrho * (1 - ux3 + 4.5 * ux2 - u215) - gridIn[i + i_w]);
      gridTemp[i + i_ne] = gridIn[i + i_ne] + omega * (diagrho * (1 + ux3 + uy3 + 4.5 * (u2 + uxuy2) - u215) - gridIn[i + i_ne]);
      gridTemp[i + i_se] = gridIn[i + i_se] + omega * (diagrho * (1 + ux3 - uy3 + 4.5 * (u2 - uxuy2) - u215) - gridIn[i + i_se]);
      gridTemp[i + i_nw] = gridIn[i + i_nw] + omega * (diagrho * (1 - ux3 + uy3 + 4.5 * (u2 - uxuy2) - u215) - gridIn[i + i_nw]);
      gridTemp[i + i_sw] = gridIn[i + i_sw] + omega * (diagrho * (1 - ux3 - uy3 + 4.5 * (u2 + uxuy2) - u215) - gridIn[i + i_sw]);
    }

    // copy densities according to motion
    fn stream(x: u32, y: u32) {
      // get indices of adjacent cells
      let i: u32 = getIndex2(x, y);
      let n: u32 = getIndex2(x, y - 1);
      let e: u32 = getIndex2(x + 1, y);
      let s: u32 = getIndex2(x, y + 1);
      let w: u32 = getIndex2(x - 1, y);
      let ne: u32 = getIndex2(x + 1, y - 1);
      let se: u32 = getIndex2(x + 1, y + 1);
      let nw: u32 = getIndex2(x - 1, y - 1);
      let sw: u32 = getIndex2(x - 1, y + 1);

      // flow and deal with boundaries
      if (x == 0 || x == ${GRID_SIZE_X - 1}) {
        // in/out boundaries
      } else if (y == 0 || y == ${GRID_SIZE_Y - 1}) {
        // up/down wall boundaries
      } else {
        // object boundaries/normal flow
        // from current to surrounding - may cause memory issues?
        gridOut[n + i_n] = gridTemp[i + i_n];
        gridOut[e + i_e] = gridTemp[i + i_e];
        gridOut[s + i_s] = gridTemp[i + i_s];
        gridOut[w + i_w] = gridTemp[i + i_w];
        gridOut[ne + i_ne] = gridTemp[i + i_ne];
        gridOut[se + i_se] = gridTemp[i + i_se];
        gridOut[nw + i_nw] = gridTemp[i + i_nw];
        gridOut[sw + i_sw] = gridTemp[i + i_sw];

        // from surroundings to current
        // gridOut[i + i_n] = gridTemp[s + i_n];
        // gridOut[i + i_e] = gridTemp[w + i_e];
        // gridOut[i + i_s] = gridTemp[n + i_s];
        // gridOut[i + i_w] = gridTemp[e + i_w];
        // gridOut[i + i_ne] = gridTemp[sw + i_ne];
        // gridOut[i + i_se] = gridTemp[nw + i_se];
        // gridOut[i + i_nw] = gridTemp[se + i_nw];
        // gridOut[i + i_sw] = gridTemp[ne + i_sw];
      }

      // compute curl
      gridOut[i + i_curl] = gridTemp[e + i_uy] - gridTemp[w + i_uy] - gridTemp[s + i_ux] + gridTemp[n + i_ux];
    }
    `
  });

  // Create a compute pipeline that updates the simulation state.
  const simulationPipeline = device.createComputePipeline({
    label: "Simulation pipeline",
    layout: pipelineLayout,
    compute: {
      module: simulationShaderModule,
      entryPoint: "computeMain",
    }
  });


  function update() {
    const encoder = device.createCommandEncoder();

    // compute pass
    const computePass = encoder.beginComputePass();

    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);

    const workgroupCountX = Math.ceil(GRID_SIZE_X / WORKGROUP_SIZE);
    const workgroupCountY = Math.ceil(GRID_SIZE_Y / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY);

    computePass.end();

    step++;

    // render pass
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
      }]
    });
    pass.setPipeline(renderPipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setBindGroup(0, bindGroups[step % 2]);
    pass.draw(vertices.length / 2, GRID_SIZE_X * GRID_SIZE_Y); // 6 vertices
    pass.end();
    device.queue.submit([encoder.finish()]);
    if (!UPDATE_INTERVAL) requestAnimationFrame(update);
    else setTimeout(update, UPDATE_INTERVAL);
    // console.log(latticeState)
    paramsArray[0] += 1;
  }
  update();
  // setInterval(update, UPDATE_INTERVAL);
}

init();