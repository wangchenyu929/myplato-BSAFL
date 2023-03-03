import matplotlib.pyplot as plt
import numpy as np

clients = np.random.uniform(1,100,50)
count, bins, ignored = plt.hist(clients, 10, density=False,facecolor='#9ac9db', edgecolor="#f8ac8c")
plt.axhline(5,linewidth=2, color='#c82423')
# plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.xlabel("Training time(s)")
plt.ylabel("Clients number")
plt.savefig("./clients_distribution.pdf")