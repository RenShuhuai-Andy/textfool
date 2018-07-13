import numpy as np
import spacy
import paraphrase

nlp = spacy.load('en')

# docs = ["i like software", "i like you"]
# docs_list = list(nlp.pipe(docs))
# print(docs_list)
# X = np.zeros((len(docs_list), 10), dtype='int32')
# for i, doc in enumerate(docs_list):
#     j = 0
#     print("doc:"+str(doc))
#     for token in doc:
#         # print("token:"+str(token)+"--token.rank:"+str(token.rank))
#         print("token:"+str(token)+"--token.tag:"+str(token.tag_)+"--type of token:"+str(type(token)))
#         if token.has_vector and not token.is_punct and not token.is_space:
#             X[i, j] = token.rank + 1
#             j += 1
#             if j >= 100:
#                 break
# print(X)


# from spacy.lang.en import English
#
# nlp = English()
# tokens = nlp('Some\nspaces  and\ttab characters')
# tokens_text = [t.text for t in tokens]
# assert tokens_text == ['Some', '\n', 'spaces', ' ', 'and',
#                        '\t', 'tab', 'characters']

docs = ["make software", "big sun"]
docs_list = list(nlp.pipe(docs))
print("docs_list: "+str(docs_list)+"--docs_list type: "+str(type(docs_list)))
for i, doc in enumerate(docs_list):
    print("doc: "+str(doc))
    paraphrase._generate_synonym_candidates(doc)
