import matplotlib.pyplot as plt
import numpy as np

# Assuming your CSV is N,ms
data = np.loadtxt("timing.csv", delimiter=',')
Ns = data[:,0]
times = data[:,1]

plt.bar([str(int(N)) for N in Ns], times)
plt.xlabel("Vector Size (N)", labelpad=20)
plt.ylabel("Execution Time (ms)", labelpad=20)
plt.title("SAXPY Execution Time by Size")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
