import os

movies_dir = 'movies'
count_movies_fi = 0
count_movies_no = 0
violent_dir = 'violentflow'
count_violent_fi = 0
count_violent_no = 0

for file in os.listdir(movies_dir):
    if "fi" in file:
        count_movies_fi += 1
    else:
        count_movies_no += 1

print(f"movies fight: {count_movies_fi}")
print(f"movies NO_fight: {count_movies_no}")

for file in os.listdir(violent_dir):
    if "violence" in file:
        count_movies_fi += 1
    else:
        count_movies_no += 1

print(f"violentflow fight: {count_movies_fi}")
print(f"violentflow NO_fight: {count_movies_no}")


