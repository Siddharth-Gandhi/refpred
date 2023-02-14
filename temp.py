# import requests
# import json
# from config import S2_API_KEY
# headers = {
#     'content-type': 'application/json',
#     'x-api-key': S2_API_KEY
# }

# # Example query
# arxiv_id = '1706.03762'
# base_url = 'https://api.semanticscholar.org/graph/v1'
# s2_search = f"{base_url}/paper/arXiv:{arxiv_id}"

# response = requests.get(s2_search, headers=headers)
# response.raise_for_status()  # raises exception when not a 2xx response
# if response.status_code == 200:
#     print(response.json())
# else:
#     print(f"Semantic Scholar did not return status 200 response")


def func():
    print(var)


if __name__ == '__main__':
    # print('hi')
    var = 'hi'
    func()
