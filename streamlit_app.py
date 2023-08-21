from newspaper import Article
import streamlit as st
import re
from transformers import PegasusForConditionalGeneration, PegasusTokenizer



model_directory = "C:/Users/moham/OneDrive/Desktop/DS Projects/NLP Project/MODEL2"

loaded_model = PegasusForConditionalGeneration.from_pretrained(model_directory)
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")


def check_input_val_is_link(input_val):
   url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
   return bool(url_pattern.match(input_val))

def return_text_for_summary(input_val):
  if check_input_val_is_link(input_val) == True:
    article_link = Article(input_val)
    try:
      article_link.download()
      article_link.parse()
      text = article_link.text
      if text == '':
        return "Unable to download article, please copy whole text into input box"
      else:
        return text
    except Exception as e:
      return "Unable to download article, please copy whole text into input box"
  else:
    return input_val
  


def count_sentences(article):
  num_sentences = 0
  for sentence in article.split('.'):
    num_sentences += 1

  return num_sentences

def generate_summary(text, model=loaded_model, tokenizer=tokenizer):
    
    if text == 'Unable to download article, please copy whole text into input box':
       return "Unable to download article, please copy whole text into input box "
    
    # Check the number of sentences in the text
    num_sentences = count_sentences(text)

    if num_sentences <= 30:
        # If the text has 30 or fewer sentences, generate a single summary
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding='max_length')
        output = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
        generated_summary = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_summary
    else:
        sentences = text.split('.') #Splitting the article on each sentence
        chunks = []
        for i in range(0, len(sentences), 30):
            chunk = " ".join(sentences[i:i + 30]) #Conjoining every 30 sentences
            chunks.append(chunk)


        chunk_summaries = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True, padding='max_length')
            output = model.generate(inputs["input_ids"], max_length=200, num_beams=4, early_stopping=True)
            chunk_summary = tokenizer.decode(output[0], skip_special_tokens=True)
            chunk_summaries.append(chunk_summary)

        chunk_summaries_conjoined = " ".join(chunk_summaries) 

        inputs = tokenizer(chunk_summaries_conjoined, return_tensors="pt", max_length=1024, truncation=True, padding='max_length')
        output = model.generate(inputs["input_ids"], max_length=200, num_beams=4, early_stopping=True)
        final_summary = tokenizer.decode(output[0], skip_special_tokens=True)

        return final_summary  


def main():
   st.title("Football Article Summarizer")

   input_val = st.text_area("Enter article link or whole text", "")

   if st.button("Generate article summary"):
      processed_output_1 = return_text_for_summary(input_val=input_val)
      if processed_output_1 != "Unable to download article, please copy whole text into input box":
         text = processed_output_1
      else:
         text = "Unable to download article, please copy whole text into input box"
    

      processed_ouput_2 = generate_summary(text=text)

      st.subheader("Article Summary")
      st.write(processed_ouput_2)

if __name__ == "__main__":
   main()
    
    