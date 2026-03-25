import numpy as np
import warp as wp
import warp.sim
import warp.sim.render


@wp.kernel
def set_p_targets(pos: wp.array(dtype=wp.vec3), 
                    initial_pos: wp.array(dtype=wp.vec3), 
                    dvf: wp.array(dtype=wp.vec3),
                    fraction: float):
    tid = wp.tid()
    pos[tid] = initial_pos[tid] + fraction * dvf[tid]

@wp.kernel
def stretch_surface_targets(pos: wp.array(dtype=wp.vec3), 
                            initial_pos: wp.array(dtype=wp.vec3), 
                            surface: wp.array(dtype=wp.float32),
                            normals: wp.array(dtype=wp.vec3),
                            fraction: float):
    tid = wp.tid()
    if surface[tid]:
        pos[tid] = initial_pos[tid] + (5.0 * fraction) * normals[tid]

@wp.kernel
def scale_restlengths(rests: wp.array(dtype=wp.float32), factor: float):
    tid = wp.tid()
    new_rest = factor * rests[tid]
    rests[tid] = new_rest

@wp.kernel
def compute_loss_kernel(
    pos: wp.array(dtype=wp.vec3),
    target_pos: wp.array(dtype=wp.vec3),
    surface: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=float),
):
    tid = wp.tid()
    # if surface[tid] < 0.5:
    tloss = wp.length(target_pos[tid] - pos[tid])
        # loss[tid] = wp.length_sq(force) # - target_pos)
    wp.atomic_add(loss, 0, tloss)

@wp.kernel(enable_backward=False)
def apply_gradient_kernel(
    spring_stiffness_grad: wp.array(dtype=float),
    train_rate: float,
    spring_stiffness: wp.array(dtype=float),
):
    tid = wp.tid()
    new_stiffness = spring_stiffness[tid] - spring_stiffness_grad[tid] * train_rate
    if new_stiffness < 0.50:
        # Prevent negative spring stiffness.
        new_stiffness = 0.50
    elif new_stiffness > 100.0:
        # Prevent too high spring stiffness.
        new_stiffness = 100.0
    spring_stiffness[tid] = new_stiffness



class ProtoCage:
    def __init__(self, connector, stage_path='usd_files/stages/cage_proto.usd', num_frames=500, train_iters=100,
                 sub_steps=100, train_rate=1.0, dvf=None):
        # Number of frames per second.
        self.fps = 60
        # Duration of a single simulation iteration in number of frames.
        self.num_frames = num_frames
        # Number of simulation steps to take per frame.
        self.sim_substep_count = sub_steps
        # Delta time between each simulation substep.
        self.sim_dt = 1.0 / (self.fps * self.sim_substep_count)
        # Target position that we want the main particle to reach by optimising
        # the rest lengths of the springs.
        self.target_pos = (0.125, -0.15, 0.375)
        # Number of training iterations.
        self.train_iters = train_iters
        # Factor by which the rest lengths of the springs are adjusted after each
        # iteration, relatively to the corresponding gradients. Lower values
        # converge more slowly but have less chances to miss the local minimum.
        self.train_rate = train_rate
        self.current_frame = 0

        # self.connector = connector
        builder = wp.sim.ModelBuilder()
        self.initial_positions = wp.from_numpy(connector.getVolumePoints(), dtype=wp.vec3f)
        self.normals = wp.from_numpy(connector.hNormals.reshape(-1,3), dtype=wp.vec3f)
        self.surface = wp.from_numpy(connector.hSurface)
        self.has_dvf = False
        if dvf is not None:
            self.has_dvf = True
            self.dvf = wp.from_numpy(dvf, dtype=wp.vec3f)
        for i, point in enumerate(connector.getVolumePoints()):
            particle_mass = connector.particle_mass
            # if connector.hSurface[i]:
            #     particle_mass = 0.0
                # point += 2*normals[i]
            builder.add_particle(pos=point, vel=(0.0, 0.0, 0.0), mass=particle_mass, radius=0.5*connector.particle_radius)
        
        spring_elastic_stiffness = 10.0
        spring_elastic_damping = 1.0
        for line in connector.hLineSet.reshape(-1,2):
            builder.add_spring(line[0], line[1], spring_elastic_stiffness, spring_elastic_damping, 0)

        self.model = builder.finalize(requires_grad=True)
        self.model.gravity = np.array((0.0, 0.0, 0.0))
        self.model.ground = False

        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.states = tuple(self.model.state() for _ in range(num_frames * self.sim_substep_count + 1))
        print(len(self.states))
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        self.renderer = warp.sim.render.SimRenderer(self.model, stage_path, fps=self.fps, scaling=10.0)
        self.render_iteration_steps = 1
        self.render_frame = 0

        self.use_cuda_graph = False #wp.get_device().is_cuda
        if self.use_cuda_graph:
            # Capture all the kernel launches into a CUDA graph so that they can
            # all be run in a single graph launch, which helps with performance.
            with wp.ScopedCapture() as capture:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(loss=self.loss)
            self.graph = capture.graph

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            # self.renderer.begin_frame(0.0)
            # self.renderer.render_box(
            #     name="target",
            #     pos=self.target_pos,
            #     rot=wp.quat_identity(),
            #     extents=(0.1, 0.1, 0.1),
            #     color=(1.0, 0.0, 0.0),
            # )
            # self.renderer.end_frame()

            for frame in range(self.num_frames):
                self.renderer.begin_frame(self.render_frame / self.fps)
                self.renderer.render(self.states[frame * self.sim_substep_count])
                self.renderer.end_frame()

                self.render_frame += 1

    def forward(self):
        self.current_frame += 1
        # fraction = float(self.current_frame % self.num_frames) / self.num_frames
        # print(self.current_frame, fraction)
        # wp.launch(kernel=set_p_targets, dim=self.model.particle_count, 
        #     inputs=(
        #         self.states[0].particle_q,
        #         self.initial_positions,
        #         self.surface,
        #         self.normals,
        #         fraction
        #     )
        # )
        # wp.launch(kernel=scale_restlengths, dim=self.model.spring_count, 
        #     inputs=(
        #         self.model.spring_rest_length, 
        #         2.0,
        #     )
        # )

        # self.particle_f: wp.array | None = None
        # """Array of 3D particle forces with shape ``(particle_count,)`` and type :class:`vec3`."""
        # last_state = None

        # compute forces
        for i in range(1, len(self.states)):   
            # if (i % self.sim_substep_count == 0):  
            self.current_frame = int(i / self.sim_substep_count)
            # print(self.current_frame)
            fraction = np.clip(float(self.current_frame) / (self.num_frames),0.0,1.0)
            # print(self.current_frame, fraction)
            if self.has_dvf:
                wp.launch(kernel=set_p_targets, dim=self.model.particle_count, 
                    inputs=(
                        self.states[i-1].particle_q,
                        self.initial_positions,
                        self.dvf,
                        fraction
                    )
                )
            else:
                wp.launch(kernel=stretch_surface_targets, dim=self.model.particle_count, 
                    inputs=(
                        self.states[i-1].particle_q,
                        self.initial_positions,
                        self.surface,
                        self.normals,
                        fraction
                    )
                )
            prev = self.states[i - 1]
            curr = self.states[i]
            prev.clear_forces()
            self.integrator.simulate(
                self.model,
                prev,
                curr,
                self.sim_dt,
            )
            # if (i == 1):
            # last_state = self.states[i]

        wp.launch(
            compute_loss_kernel,
            dim=1,
            inputs=(
                self.states[-1].particle_q,
                self.initial_positions,
                self.surface
            ),
            outputs=(self.loss,),
        )

        # wp.launch(kernel=scale_restlengths, dim=self.model.spring_count, 
        #     inputs=(
        #         self.model.spring_rest_length, 
        #         0.5,
        #     )
        # )
        
    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(loss=self.loss)

            wp.launch(
                apply_gradient_kernel,
                dim=self.model.spring_count,
                inputs=(
                    self.model.spring_stiffness.grad,
                    self.train_rate,
                ),
                outputs=(self.model.spring_stiffness,),
            )

            # if self.verbose:
            print("Spring stiffness:", self.model.spring_stiffness)
            #     print("Spring stiffness gradient:", self.model.spring_stiffness.grad)

            self.tape.zero()

    def get_average_stiffness_per_particles(self):
        """
        For each particle, compute the average spring stiffness of all springs connected to it,
        and localize the result in a 3D volume using index_volume.
        """
        # Accumulate stiffness and counts for each particle
        stiffness_sum = np.zeros(self.model.particle_count, dtype=np.float32)
        stiffness_count = np.zeros(self.model.particle_count, dtype=np.int32)

        spring_ks = self.model.spring_stiffness.numpy()
        for i, (a, b) in enumerate(self.model.spring_indices.numpy().reshape(-1,2)):
            stiffness_sum[a] += spring_ks[i]
            stiffness_sum[b] += spring_ks[i]
            stiffness_count[a] += 1
            stiffness_count[b] += 1

        # Avoid division by zero
        avg_stiffness = np.zeros_like(stiffness_sum)
        mask = stiffness_count > 0
        avg_stiffness[mask] = stiffness_sum[mask] / stiffness_count[mask]

        return avg_stiffness