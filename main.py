import requests, re

def main():
    try:
        dict = {}
        r = requests.get("https://www.gutenberg.org/browse/scores/top")
        top100 = re.findall(r"/ebooks/(\d+)", r.text)
        print(top100)
        odyssey = "https://www.gutenberg.org/cache/epub/" + top100[0] + "/pg" + top100[0] + "-images.html"

        t = requests.get(odyssey)
        print(repr(t.text[90000:92000]))

        #odyssey_pages = t.text[90000:100000].split("\n")
        odyssey_pages = re.findall(r"<p>.*?</p>", t.text[90000:92000], flags=re.DOTALL)
        print(odyssey_pages)

        return 0
    
    except Exception as e:
        print("An error occurred: " + e)
        return 1

if __name__ == "__main__":
    main()