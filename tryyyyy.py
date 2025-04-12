import http.client

conn = http.client.HTTPSConnection("exercisedb-api.vercel.app")

conn.request("GET", "/api/v1/exercises")

res = conn.getresponse()
data = res.read()

print(data)
print(data.decode("utf-8"))