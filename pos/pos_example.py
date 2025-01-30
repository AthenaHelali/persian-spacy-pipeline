import spacy

# Load the trained POS model
model_path = "/persian_spacy/models/pos"
nlp = spacy.load(model_path)

# Sentences with diverse POS tags
sentences = [
    "امروز، با وجود هوای بارانی، بچه‌ها در پارک با دوستانشان بازی کردند و حسابی خوش گذراندند.",
    "این کتاب ارزشمند، که درباره تاریخ تمدن‌های باستان نوشته شده، جایزه بهترین اثر سال را دریافت کرده است.",
    "اگرچه ماشین جدید سریع و زیبا است، اما مصرف سوخت آن بیشتر از مدل قبلی است.",
    "استاد دانشگاه، پس از پایان سخنرانی جذاب خود، به سوالات دانشجویان پاسخ داد.",
    "در این فصل از سال، طبیعت سرسبز و زیبایی خیره‌کننده‌ای دارد که دل هر بیننده‌ای را می‌برد.",
    "برای موفقیت در این پروژه، همکاری دقیق و منظم بین اعضای تیم الزامی است.",
    "مریم، که برای مسابقه آماده می‌شود، تمرینات سختی را زیر نظر مربی انجام می‌دهد.",
]

# Process and display sentences
for sentence in sentences:
    doc = nlp(sentence)
    print(f"Sentence: {sentence}")
    print("Token\tPOS Tag")
    print("----------------")
    for token in doc:
        print(f"{token.text}\t{token.tag_}")
