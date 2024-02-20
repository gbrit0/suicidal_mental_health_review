# Suicidal Mental Health Review
Implementação de algoritmo de machine learning para análise de texto visando identificar intenções suicididas e depressão. Desenvolvido apenas para fins de estudo tecnológico sem a intenção de provêr aconselhamento psicológico ou médico.

Este código utiliza os dados do dataset [Suicidal Mental Health Review](https://www.kaggle.com/datasets/subhranilpaul/suicidal-mental-health-review) disponível no Kaggle sob a licença MIT.

Ademais, utilizam-se bibliotecas desenvolvidas ao longo do livro Data Science do Zero - Noções fundamentais com Python de Joel Grus disponibilizadas em seu repositório [github](https://github.com/joelgrus/data-science-from-scratch/tree/master) também sob licença MIT.

### O Dataset
Arquivo .csv com duas colunas: a primeira contendo um texto no qual os participantes da Mental Health Week descreveram como se sentiam no momento; e a segunda rotulando o texto em 'depression' ou 'SuicideWatch'. Foram coletadas e rotuladas 20339 respostas.

### O Modelo
Por se tratar de análise de texto, optou-se por transforma-los em palavras distintas, tokens. Os textos foram reformulados em letras minúsculas, "palavras" formadas por letras únicas, números e apóstrofos ou pontuação, foram removidas. Um Set recebeu as palavras distintas.

Foram definidas classes para as mensagens e para o classificador Naive Bayes, que foi o adotado.

