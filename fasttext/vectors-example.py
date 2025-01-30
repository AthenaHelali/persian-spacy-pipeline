import spacy

nlp = spacy.blank("fa")
nlp.vocab.vectors.from_disk("persian_spacy/fasttext/vocab")
print(nlp.vocab.vectors.shape)

token1 = nlp("ایران")[0]
token2 = nlp("کشور")[0]

# Calculate similarity between the two tokens
similarity = token1.similarity(token2)
print(f"Token Similarity: {similarity}")


sentence1 = nlp("ایران یک کشور در خاورمیانه است.")
sentence2 = nlp("کشور ایران در آسیا قرار دارد.")

# Calculate sentence similarity
similarity = sentence1.similarity(sentence2)
print(f"Sentence Similarity: {similarity}")

sentence1 = nlp(
    "الگوریتم‌های جستجو در موتورهای جستجو برای یافتن اطلاعات آنلاین استفاده می‌شوند."
)
sentence2 = nlp("شکوفایی هنرهای تجسمی در دهه‌های اخیر به شدت تحت تاثیر جوانان بوده است.")

# Calculate sentence similarity
similarity = sentence1.similarity(sentence2)
print(f"Sentence Similarity: {similarity}")
