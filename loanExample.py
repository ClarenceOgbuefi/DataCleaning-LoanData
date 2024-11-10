import numpy as np
import pandas as pd


# Checkpoint Save
def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header=checkpoint_header, data=checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return checkpoint_variable


# Import Raw Data
np.set_printoptions(suppress=True, linewidth=100, precision=2)

raw_data_np = np.genfromtxt("loan-data.csv",
                            delimiter=";",
                            encoding="ISO-8859-1",
                            skip_header=1,
                            autostrip=True)

print(raw_data_np)

# Checking for Incomplete Data
print(np.isnan(raw_data_np).sum())

temporary_fill = np.nanmax(raw_data_np) + 1
temporary_mean = np.nanmean(raw_data_np, axis=0)
print(temporary_mean)

temporary_stats = np.array([np.nanmin(raw_data_np, axis=0),
                            np.nanmean(raw_data_np, axis=0),
                            np.nanmax(raw_data_np, axis=0)])
print(temporary_stats)

# Splitting the Dataset
columns_strings = np.argwhere((np.isnan(temporary_mean))).squeeze()
print(columns_strings)

columns_numeric = np.argwhere((np.isnan(temporary_mean) == False)).squeeze()
print(columns_numeric)

loan_data_strings = np.genfromtxt("loan-data.csv",
                                  delimiter=";",
                                  encoding="ISO-8859-1",
                                  skip_header=1,
                                  autostrip=True,
                                  usecols=columns_strings,
                                  dtype=str)

loan_data_numeric = np.genfromtxt("loan-data.csv",
                                  delimiter=";",
                                  encoding="ISO-8859-1",
                                  skip_header=1,
                                  autostrip=True,
                                  usecols=columns_numeric,
                                  filling_values=temporary_fill)

print(loan_data_numeric)

print(loan_data_strings)
print(raw_data_np.shape[0])

# Pull the names of columns
header_full = np.genfromtxt("loan-data.csv",
                            delimiter=";",
                            encoding="ISO-8859-1",
                            skip_footer=raw_data_np.shape[0],
                            autostrip=True,
                            dtype=str)
print(header_full)
header_string, header_numeric = header_full[columns_strings], header_full[columns_numeric]

# Manipulating String Columns
print(header_string)
header_string[0] = "issue_date"
print(header_string)

print(loan_data_strings)
print(np.unique(loan_data_strings[:, 0]))
loan_data_strings[:, 0] = np.char.strip(loan_data_strings[:, 0], "-15")
print(loan_data_strings)

months = np.array(["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
for i in range(13):
    loan_data_strings[:, 0] = np.where(loan_data_strings[:, 0] == months[i],
                                       str(i),
                                       loan_data_strings[:, 0])
print(loan_data_strings[:, 0])

# Loan Status
print(header_string)
print(np.unique(loan_data_strings[:, 1]))
status_good = np.array(["Current", "Fully Paid", "In Grace Period", "Issued", "Late (16-30 days)"])
loan_data_strings[:, 1] = np.where(np.isin(loan_data_strings[:, 1], status_good), 1, 0)
print(np.unique(loan_data_strings[:, 1]))

# Term
print(header_string)
print(np.unique(loan_data_strings[:, 2]))
loan_data_strings[:, 2] = np.char.strip(loan_data_strings[:, 2], " months")
print(loan_data_strings[:, 2])
header_string[2] = "term_months"
loan_data_strings[:, 2] = np.where(loan_data_strings[:, 2] == "",
                                   "60",
                                   loan_data_strings[:, 2])
print(np.unique(loan_data_strings[:, 2]))

# Grade and Subgrade
print(header_string)
print(np.unique(loan_data_strings[:, 3]))
print(np.unique(loan_data_strings[:, 4]))
print(type(np.unique(loan_data_strings[:, 3][1])))
for i in np.unique(loan_data_strings[:, 3])[1:]:
    print(type(i))
    loan_data_strings[:, 4] = np.where((loan_data_strings[:, 4] == "") & (loan_data_strings[:, 3] == i),
                                       i + "5",
                                       loan_data_strings[:, 4])
print(np.unique(loan_data_strings[:, 4], return_counts=True))
loan_data_strings[:, 4] = np.where(loan_data_strings[:, 4] == "",
                                   "H1",
                                   loan_data_strings[:, 4])
print(np.unique(loan_data_strings[:, 4]))

# Removing Grade
loan_data_strings = np.delete(loan_data_strings, 3, axis=1)
print(loan_data_strings[:, 3])
header_string = np.delete(header_string, 3)
print(header_string[3])

# Converting Subgrade
keys = list(np.unique(loan_data_strings[:, 3]))
values = list(range(1, loan_data_strings[:, 3].shape[0] + 1))
dict_sub_grade = dict(zip(keys, values))

for number in np.unique(loan_data_strings[:, 3]):
    loan_data_strings[:, 3] = np.where(loan_data_strings[:, 3] == number,
                                       str(dict_sub_grade[number]),
                                       loan_data_strings[:, 3])
print(loan_data_strings[:, 3])

# Verification Status
print(header_string[4])
print(np.unique(loan_data_strings[:, 4]))

status_good = np.array(["Source Verified", "Verified"])
loan_data_strings[:, 4] = np.where(np.isin(loan_data_strings[:, 4], status_good), 1, 0)
print(loan_data_strings[:, 4])

# URL
print(header_string[5])
print(np.unique(loan_data_strings[:, 5]))

loan_data_strings[:, 5] = np.char.strip(loan_data_strings[:, 5],
                                        "https://www.lendingclub.com/browse/loanDetail.action?loan_id=")
print(loan_data_strings[:, 5])

print(loan_data_numeric[:, 0])
print(np.array_equal(loan_data_numeric[:, 0], loan_data_strings[:, 5].astype(dtype=int)))

loan_data_strings = np.delete(loan_data_strings, 5, axis=1)
header_string = np.delete(header_string, 5)
print(loan_data_strings)
print(header_string)

# State Addresses
states_names, states_count = np.unique(loan_data_strings[:, 5], return_counts=True)
states_count_sorted = np.argsort(-states_count)
print(states_names[states_count_sorted], states_count[states_count_sorted])

# Account for Missing States
loan_data_strings[:, 5] = np.where(loan_data_strings[:, 5] == "",
                                   "0",
                                   loan_data_strings[:, 5])
# Organize States by Region
states_west = np.array(['WA', 'OR', 'CA', 'NV', 'ID', 'MT', 'WY', 'UT', 'CO', 'AZ', 'NM', 'HI', 'AK'])
states_south = np.array(['TX', 'OK', 'AR', 'LA', 'MS', 'AL', 'TN', 'KY', 'FL', 'GA', 'SC', 'NC', 'VA', 'WV', 'MD', 'DE', 'DC'])
states_midwest = np.array(['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH'])
states_east = np.array(['PA', 'NY', 'NJ', 'CT', 'MA', 'VT', 'NH', 'ME', 'RI'])

loan_data_strings[:, 5] = np.where(np.isin(loan_data_strings[:, 5], states_west), "1", loan_data_strings[:, 5])
loan_data_strings[:, 5] = np.where(np.isin(loan_data_strings[:, 5], states_south), "2", loan_data_strings[:, 5])
loan_data_strings[:, 5] = np.where(np.isin(loan_data_strings[:, 5], states_midwest), "3", loan_data_strings[:, 5])
loan_data_strings[:, 5] = np.where(np.isin(loan_data_strings[:, 5], states_east), "4", loan_data_strings[:, 5])

print(np.unique(loan_data_strings[:, 5]))

# Converting to Numbers
print(loan_data_strings)
loan_data_strings = loan_data_strings.astype(dtype=int)
print(loan_data_strings)

# Checkpoint
checkpoint_strings = checkpoint("Checkpoint-Strings", header_string, loan_data_strings)

# Checking for Missing Values
print(loan_data_numeric)
print(np.isnan(loan_data_numeric).sum())

print(np.isin(loan_data_numeric[:, 0], temporary_fill).sum())

print(temporary_stats[:, columns_numeric])
print(temporary_stats[:, columns_numeric[2]])

# Fill in missing values
loan_data_numeric[:, 2] = np.where(loan_data_numeric[:, 2] == temporary_fill,
                                   temporary_stats[0, columns_numeric[2]],
                                   loan_data_numeric[:, 2])
print(loan_data_numeric[:, 2])

for i in [1, 3, 4, 5]:
    loan_data_numeric[:, i] = np.where(loan_data_numeric[:, i] == temporary_fill,
                                       temporary_stats[2, columns_numeric[i]],
                                       loan_data_numeric[:, i])


# Currency Change
EUR_USD = np.genfromtxt("EUR-USD.csv",
                        delimiter=",",
                        autostrip=True,
                        skip_header=1,
                        usecols=[3])
print(EUR_USD)
print(loan_data_strings)
exchange_rate = loan_data_strings[:, 0]

for i in range(1, 13):
    exchange_rate = np.where(exchange_rate == i,
                             EUR_USD[i - 1],
                             exchange_rate)

exchange_rate = np.where(exchange_rate == 0,
                         np.mean(EUR_USD),
                         exchange_rate)
print(exchange_rate)
print(exchange_rate.shape)
print(loan_data_numeric.shape)

exchange_rate = np.reshape(exchange_rate, (10000, 1))
loan_data_numeric = np.hstack((loan_data_numeric, exchange_rate))
print(loan_data_numeric)
header_numeric = np.concatenate((header_numeric, np.array(["exchange_rate"])))
print(header_numeric)

# Currency Exchange
print(header_numeric)
column_dollar = np.array([1, 2, 4, 5])
print(loan_data_numeric[:, 6])
for i in column_dollar:
    loan_data_numeric = np.hstack((loan_data_numeric,
                                   np.reshape(loan_data_numeric[:, i] / loan_data_numeric[:, 6], (10000, 1))))
print(loan_data_numeric.shape)

header_additional = np.array([column_name + "_EUR" for column_name in header_numeric[column_dollar]])
print(header_additional)
header_numeric = np.concatenate((header_numeric, header_additional))

header_numeric[column_dollar] = np.array([column_name + "_USD" for column_name in header_numeric[column_dollar]])
print(header_numeric)

# Rearrange columns
columns_index_order = [0, 1, 7, 2, 8, 3, 4, 9, 5, 10, 6]
header_numeric = header_numeric[columns_index_order]
print(header_numeric)

loan_data_numeric = loan_data_numeric[:, columns_index_order]
print(loan_data_numeric)

# Interest Rate Conversion to Decimal
print(header_numeric)
print(loan_data_numeric[:, 5])
loan_data_numeric[:, 5] = loan_data_numeric[:, 5]/100
print(loan_data_numeric[:, 5])

# Checkpoint 2
checkpoint_numeric = checkpoint("Checkpoint-Numeric", header_numeric, loan_data_numeric)

# Combine the data
print(checkpoint_strings["data"].shape)
print(checkpoint_numeric["data"].shape)
loan_data = np.hstack((checkpoint_numeric["data"], checkpoint_strings["data"]))
print(np.isnan(loan_data).sum())

# Combine the Headers
loan_header = np.concatenate((checkpoint_numeric["header"], checkpoint_strings["header"]))
print(loan_header)

# Sorting the Dataset
loan_data = loan_data[np.argsort(loan_data[:, 0])]
print(loan_data)

# Storing the New Dataset
loan_data_full = np.vstack((loan_header, loan_data))
np.savetxt("loan-data-preprocessed.csv",
           loan_data_full,
           fmt="%s",
           delimiter=",")


