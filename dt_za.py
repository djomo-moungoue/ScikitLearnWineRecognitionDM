from datetime import datetime, timedelta, date, time

print(f"Now : {datetime.now()}")
# Résultat : Now : 2022-10-24 20:48:48.486493
print(f'Datetime object : {datetime(2022, 10, 25)}')
# Résultat : Datetime object : 2022-10-25 00:00:00
user_input = "2018/01/01"
dt = datetime.fromtimestamp(time.time())
print(f'time converted to datetime object : {dt}')
# Résultat : time converted to datetime object : 2022-10-24 20:48:48.486494
print(f'string converted to datetime object : {datetime.strptime(user_input, "%Y/%m/%d")}')
# Résultat : string converted to datetime object : 2018-01-01 00:00:00
print(f'datetime object converted to string : {dt.strftime("%Y/%m/%d")}')
# Résultat : datetime object converted to string : 2022/10/24
print(f'datetime object attributes : {dt.year}-{dt.month}')
# Résultat : datetime object attributes : 2022-10