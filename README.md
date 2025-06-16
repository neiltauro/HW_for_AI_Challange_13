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

## 🛠 Installation and Compilation:

✅ **Requirements:**

- CUDA Toolkit installed
- `nvcc` compiler available in your PATH
- (Optionally) Python with matplotlib for visualization

✅ **Compile with:**

```bash
nvcc saxpy.cu -o saxpy

## 🔹 Conclusion and Findings 🔹

In this benchmark, we successfully implemented a **SAXPY kernel with CUDA** and measured its execution time across a range of vector sizes.  
We observed:

- The execution time grows **with the vector size**, reflecting the additional computational and memory workloads.
- The GPU efficiently handles large workloads due to its massive parallelism, yielding faster per-element processing.
- The main bottleneck for performance is often **global memory transfer** and not the kernel execution itself.
- Proper selection of **thread block sizes** can aid performance but typically, the main scaling comes from the algorithm’s ability to leverage GPU parallelism.

Overall, this benchmark illustrates the power of CUDA for data-parallel workloads and emphasizes the importance of understanding both kernel execution and memory transfer mechanisms for optimizing performance.
