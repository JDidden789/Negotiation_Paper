import random
import numpy as np
import numpy.random
import pandas as pd

operationOrder = [
    [1, 5, 3, 6],
    [2, 6, 5, 4, 3, 1],
    [3, 2, 4],
    [5, 4, 3, 1, 6, 2],
    [2, 6, 3, 5, 1, 4],
    [5, 2, 6, 3, 1],
    [3, 2, 4, 1],
    [4, 5, 3, 1, 6, 2],
    [6, 1, 4, 2, 3],
    [1, 4, 2, 6],
    [5, 6, 1, 3, 2, 4],
    [4, 5, 1],
    [5, 3, 1, 2],
    [5, 4, 2, 1, 3],
    [2, 3, 5, 1, 4, 6],
    [2, 4, 6, 1, 5, 3],
    [3, 4, 1, 5, 6, 2],
    [2, 1, 5],
    [5, 2, 1],
    [2, 5, 1],
]

demand = [
    0.026392961876832845,
    0.09286412512218964,
    0.004887585532746823,
    0.05180840664711633,
    0.06940371456500488,
    0.04887585532746823,
    0.0967741935483871,
    0.05767350928641251,
    0.0009775171065493646,
    0.002932551319648094,
    0.007820136852394917,
    0.08504398826979472,
    0.07917888563049853,
    0.06647116324535679,
    0.07038123167155426,
    0.03128054740957967,
    0.07624633431085044,
    0.01857282502443793,
    0.024437927663734114,
    0.08797653958944282,
]
priority_per_job = [
    2.6224272,
    1.17527717,
    5.16896674,
    7.52440536,
    4.78183244,
    5.36884388,
    1.11502733,
    5.38634447,
    9.47625987,
    8.6571558,
    7.56968023,
    1.97862465,
    9.04513753,
    8.71438822,
    2.48577956,
    6.69100612,
    1.18435252,
    2.05063542,
    3.8473058,
    2.42121076,
]

# processingTimes1 = [
#     [30, 10, 18, 25],
#     [30, 18, 17, 10, 12, 31],
#     [15, 18, 14],
#     [22, 12, 20, 18, 14, 27],
#     [22, 28, 18, 29, 10, 17],
#     [28, 17, 26, 22, 30],
#     [11, 20, 10, 34],
#     [32, 13, 19, 27, 10, 23],
#     [16, 19, 23, 15, 30],
#     [13, 10, 12, 29],
#     [29, 32, 14, 13, 23, 25],
#     [20, 29, 25],
#     [26, 27, 22, 10],
#     [18, 23, 24, 34, 20],
#     [18, 12, 34, 10, 21, 22],
#     [20, 11, 19, 15, 29, 31],
#     [33, 28, 15, 26, 25, 27],
#     [22, 29, 34],
#     [11, 29, 28],
#     [29, 32, 26],
# ]

# setupTime1 = pd.read_csv("./Old/Setup_Time_New.csv", index_col=0)
# setupTime1 = setupTime1.values.tolist()


def new_scenario(max_workcenters, min_workcenters, no_of_jobs, total_machines, min_proc, max_proc, setup_factor):
    np_seed = numpy.random.seed(150)

    processingTime = []
    # for i in operations_per_job:
    #     operationOrder.append(list(np.random.choice(range(1, max_workcenters + 1), i, replace=False)))
    operations_per_job = [len(i) for i in operationOrder]
    if max_proc != 35:
        # priority_per_job = np.random.uniform(low=1.0, high=10.0, size=no_of_jobs)
        # operations_per_job = np.random.choice(range(min_workcenters, max_workcenters + 1), no_of_jobs)
        # demand = np.random.choice(range(10), no_of_jobs)
        # demand = [x / sum(demand) for x in demand]

        processingTime.extend(
            list(np.random.randint(min_proc, max_proc, operations_per_job[i])) for i in range(no_of_jobs)
        )
    else:
        processingTime = [
            [30, 10, 18, 25],
            [30, 18, 17, 10, 12, 31],
            [15, 18, 14],
            [22, 12, 20, 18, 14, 27],
            [22, 28, 18, 29, 10, 17],
            [28, 17, 26, 22, 30],
            [11, 20, 10, 34],
            [32, 13, 19, 27, 10, 23],
            [16, 19, 23, 15, 30],
            [13, 10, 12, 29],
            [29, 32, 14, 13, 23, 25],
            [20, 29, 25],
            [26, 27, 22, 10],
            [18, 23, 24, 34, 20],
            [18, 12, 34, 10, 21, 22],
            [20, 11, 19, 15, 29, 31],
            [33, 28, 15, 26, 25, 27],
            [22, 29, 34],
            [11, 29, 28],
            [29, 32, 26],
        ]
    # workload_per_wc = np.zeros(max_workcenters)
    # for i in range(no_of_jobs):
    #     for count, value in enumerate(operationOrder[i]):
    #         workload_per_wc[value - 1] += processingTime[i][count] * demand[i]

    # division_per_workcenter = [x / sum(workload_per_wc) for x in workload_per_wc]
    # machines_per_wc = [round(x * total_machines) for x in division_per_workcenter]
    # machine_number_WC = []

    # k = 1
    # for i in range(max_workcenters):
    #     machine_number_WC.append(list(range(k, machines_per_wc[i] + k)))
    #     k += machines_per_wc[i]

    # Set setup times
    if setup_factor != 0.3:
        max_setup = setup_factor * 22.5
        b = np.random.uniform(0, max_setup, size=(no_of_jobs, no_of_jobs))
        setupTime = (b + b.T) / 2
        for i in range(no_of_jobs):
            setupTime[i][i] = 0
    else:
        setupTime = pd.read_csv("./Old/Setup_Time_New.csv", index_col=0)
        setupTime = setupTime.values.tolist()

    # setupTime = pd.DataFrame(setupTime)
    # file_name = f"Setup_Time_0.5.csv"
    # setupTime.to_csv(file_name)

    # Calculate arrival rate
    utilization = [0.9]
    # job_in_wc = []
    # mean_setup = []
    # for j in range(no_of_jobs):
    #     mean_setup.append(list(np.zeros(len(operationOrder[j]))))

    # for w in range(max_workcenters):
    #     job_in_wc.append([])
    #     for j in range(no_of_jobs):
    #         if (w + 1) in operationOrder[j]:
    #             job_in_wc[w].append([j, operationOrder[j].index(w + 1)])

    # for i in job_in_wc:
    #     # mean_setup.append([])
    #     for j in i:
    #         for k in i:
    #             mean_setup[j[0]][j[1]] += setupTime[j[0]][k[0]] * demand[k[0]]

    mean_processing_time = sum([np.mean(processingTime[i]) * demand[i] for i in range(no_of_jobs)])
    # mean_setup_time = sum([np.mean(mean_setup[i]) * demand[i] for i in range(no_of_jobs)])
    mean_operations_per_job = sum(operations_per_job[i] * demand[i] for i in range(no_of_jobs))
    arrival_rate = [mean_processing_time * mean_operations_per_job / (total_machines * i) for i in utilization]
    # print(arrival_rate)
    # sumProcessingTime = [sum(processingTime[i]) for i in range(no_of_jobs)]

    # # Get maximum critical ratio
    # max_ddt = 8
    # CR = []
    # DDT = []
    # for j in range(no_of_jobs):
    #     CR.append([(sumProcessingTime[j] * max_ddt - sum(processingTime[j][0:i])) / sum(processingTime[j][i:]) for i in
    #                range(operations_per_job[j] - 1)])
    #     DDT.append(sumProcessingTime[j] * max_ddt)

    machinesPerWC = [5, 4, 3, 3, 4, 2]
    mean_processing_time = np.zeros(len(machinesPerWC))
    total_per_demand = np.zeros(len(machinesPerWC))
    for j, job in enumerate(operationOrder):
        for operation in job:
            total_per_demand[int(operation) - 1] += demand[j]

    for j, job in enumerate(operationOrder):
        for i, operation in enumerate(job):
            mean_processing_time[int(operation) - 1] += (
                processingTime[j][i] * demand[j] / total_per_demand[int(operation) - 1]
            )

    # workload_per_wc = [
    #     23.73216031,
    #     20.12903226,
    #     15.11241447,
    #     13.0459433,
    #     22.05962854,
    #     11.24731183,
    # ]
    # totalAttributes = max([noAttributes + noAttributesJob, noAttributesJRA, noAttributesNeg])
    bounded_values = [
        [-max(max(processingTime)), max(max(processingTime))],
        [-800, 800],
        [-800, 800],
        [-10, 10],
        [-308, 308],
        [0, 12],
        [0, 500],
        [0, 2],
        [0, 6 * np.max(mean_processing_time)],
    ]

    return processingTime, setupTime, mean_processing_time, bounded_values


# if __name__ == "__main__":
#     (processingTime, setupTime, mean_processing_time, bounded_values) = new_scenario(6, 3, 20, 21, 10, 35, 0.5)
# print(setupTime)
