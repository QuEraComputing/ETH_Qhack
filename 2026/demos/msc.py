import tsim
import time

c = tsim.Circuit.from_file("../circuits/msc_3.stim")


# Compilation step, takes about 3s
sampler = c.compile_detector_sampler(strategy="cutting")

start_time = time.perf_counter()
n = 10_000  # On GPU, increase batch size to fully utilize VRAM
sampler.sample(shots=n, batch_size=n)
end_time = time.perf_counter()
print(f"Time per shot: {(end_time - start_time) / n * 1e6:.2f} microseconds")
