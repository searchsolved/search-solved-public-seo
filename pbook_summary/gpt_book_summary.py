import openai
import re
import base64
import requests
import pandas as pd

openai.api_key = "YOUR_KEY_HERE"
# wp credentials

user = "WP LOGIN"
password = "ENCODED PASSWORD"
url = "URL"

df = pd.read_csv('/python_scripts/book_summary/business/top_100_business_books.csv')
total_books = len(df)
count = 1

for book in df['title']:
  print("Processing:", count, "of", total_books, "\nTitle:", book)

  try:
    # -------------------------

    input = "author for the book (concise as possible): " + str(book)

    resp = openai.Completion.create(
        model="text-davinci-003",
        prompt=input,
        max_tokens=250,
        temperature=0,

    )

    author = (resp.choices[0].text)
    print(author)

    input = "name of publisher for the book (as concise as possible): " + str(book) + "by" + author

    resp = openai.Completion.create(
        model="text-davinci-003",
        prompt=input,
        max_tokens=100,
        temperature=0,

    )

    publisher = (resp.choices[0].text)
    print(publisher)

    input = "extensive bullet list summary (<ul tags>) of key points for the book: " + str(book) + "by" + author

    resp = openai.Completion.create(
        model="text-davinci-003",
        prompt=input,
        max_tokens=1000,
        temperature=0,

    )

    summary = (resp.choices[0].text)

    # ----------------------

    # -----------------
    input = "write a conclusion for the book: " + str(book) + "by" + author

    resp = openai.Completion.create(
        model="text-davinci-003",
        prompt=input,
        max_tokens=1500,
        temperature=0,

    )

    conclusion = (resp.choices[0].text)

    # -----------------
    input = "write a critical review for the book: " + str(book) + "by" + author

    resp = openai.Completion.create(
        model="text-davinci-003",
        prompt=input,
        max_tokens=1500,
        temperature=0,

    )

    review = (resp.choices[0].text)
    print(review)

    input = "write some pros and cons(<ul> tags, pro and cons heading in <h2>) for the book: " + str(book)

    resp = openai.Completion.create(
        model="text-davinci-003",
        prompt=input,
        max_tokens=1500,
        temperature=0,

    )

    pros = (resp.choices[0].text)
    print(pros)

    # -------------

    input = "suggest up to three category tags for the book:" + str(book)

    resp = openai.Completion.create(
        model="text-davinci-003",
        prompt=input,
        max_tokens=1000,
        temperature=0,

    )
    tags = (resp.choices[0].text)

    # put it all together

    doc = (
            "<strong>"
            + book
            + "</strong>"
            + "\n"
            + "<h2>Shortform Summary</h2>"
            + "\n"
            + summary
            + "\n"
            + "<h2>Review</h2>"
            + review
            + "\n"
            + pros
            + "\n"
            + "<h2>Conclusion</h2>"
            + conclusion
            + "\n"
            + author
            + publisher
            + "\n"
            + "\n"
            + "Tags: "
            + tags
    )

    # cleanup the file name, remove special charcs
    file_name = book
    file_name = re.sub(r"[^A-Za-z]+", ' ', file_name)

    f = open("/python_scripts/book_summary/business/" + str(file_name) + ".txt", "w")

    # Write a string to the file
    f.write(doc)

    # Close the file
    f.close()

    # ----- save to wp

    # encode the connection of your WordPress website
    wp_connection = user + ':' + password
    token = base64.b64encode(wp_connection.encode())

    # prepare the header of our request
    headers = {'Authorization': 'Basic ' + token.decode('utf-8')}

    # define a title for our first post
    post_title = book

    # define a body for our post
    post_body = doc

    # then we need to set the type of our post and assign the content values to it using a Python dictionary
    post = {'title': post_title,
            'status': 'publish',
            'content': post_body,
            'author': '1',
            'format': 'standard',
            }

    # finally, we will perform the request on the REST API to insert the post.
    wp_request = requests.post(url + '/posts', headers=headers, json=post)
  except Exception:
    print("Error! Skipping:", book)
    count = count + 1
  count = count+1
