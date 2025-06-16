# SAXPY CUDA Benchmark Challenge

## 🔹 Challenge #13 — Benchmarking different SAXPY problem sizes 🔹

This project is a benchmark of **SAXPY (Single-Precision A∙X Plus Y)** implemented in CUDA.  
It measures execution time for different vector sizes and shows their performance trends.

---

## 📚 Challenge Goals:

- Set up a CUDA development environment.
- Modify existing CUDA code to enable execution with multiple vector sizes.
- Measure execution time using CUDA events.
- Visualize the results in a bar plot.

---

## 🔹 What We Observe 🔹

- The **CPU version executes faster** for small input sizes due to minimal overhead and direct sequential computation.

- The **GPU kernel underperforms** for this Fibonacci implementation because:

  - Each Fibonacci number depends directly on its two preceding values, creating a **serial dependency**.

  - This dependency chain **prevents efficient parallelization**, which limits the advantages of GPU parallelism.

- GPUs excel at large-scale parallel tasks with **independent computations**, but Fibonacci’s recursive data dependency makes it a poor fit for parallel GPU acceleration.

- For algorithms with inherent sequential dependencies like Fibonacci, **CPU implementations often outperform GPU ones**, especially for smaller problem sizes.

- This highlights the importance of **choosing the right computational platform based on the problem structure**.

--

## 📚 Findings:

- The execution time grows **with the vector size**, reflecting the additional computational and memory workloads.
- The GPU efficiently handles large workloads due to its massive parallelism, yielding faster per-element processing.
- The main bottleneck for performance is often **global memory transfer** and not the kernel execution itself.
- Proper selection of **thread block sizes** can aid performance but typically, the main scaling comes from the algorithm’s ability to leverage GPU parallelism.


---

## 🛠 Installation and Compilation:

✅ **Requirements:**

- CUDA Toolkit installed
- `nvcc` compiler available in your PATH
- (Optionally) Python with matplotlib for visualization

✅ **Compile with:**

```bash
nvcc saxpy.cu -o saxpy
