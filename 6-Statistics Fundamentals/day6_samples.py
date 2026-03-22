import scipy.stats as stats


data = [10, 20, 30, 40, 50]
mean = sum(data) / len(data)

variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
std_dev = variance ** 0.5

sample_mean = mean
t_crit = stats.t.ppf(0.975, df=len(data) - 1)

ci = (sample_mean - t_crit * (std_dev / (len(data) ** 0.5)),
      sample_mean + t_crit * (std_dev / (len(data) ** 0.5)))

if __name__ == "__main__":
    print("95% Confidence Interval:", ci) 
# from statistics import mode

# data = [10, 20, 30, 40, 50]
# mean = sum(data) / len(data)
# print("Mean: ", mean)

# sorted_data = sorted(data)
# median = sorted_data[len(data) // 2] if len(data) % 2 != 0 else \
#     (sorted_data[len(data) // 2 - 1] + sorted_data[len(data) // 2]) / 2
# print("Median: ", median)

# print("Mode: ", mode(data))

# variance = sum((x - mean) ** 2 for x in data) / len(data)
# print("Variance: ", variance)
# std_dev = variance ** 0.5
# print("Standard Deviation:", std_dev)