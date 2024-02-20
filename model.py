# Suicidal Mental Health Review
# Este código utiliza dados do dataset Suicidal Mental Health Review disponível no Kaggle sob a licença MIT.
# Baseado nas funções desenvolvidas ao longo do livro Data Science do Zero - Noções fundamentais com Python de Joel Grus

from typing import Set
import re

def tokenize(text: str) -> Set[str]:
   text = text.lower()                    # Converta para minúsculas
   all_words = re.findall("[a-z0-9']+", text)
   return set(all_words)

assert tokenize("Health Review. Suicidal mental health review") == {'suicidal', 'review', 'health', 'mental'}

# Definição de uma classe para os dados de treinamento
from typing import NamedTuple

class Message(NamedTuple):
   text: str
   suicidewatch: bool = False   

from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:
   def __init__(self, k: float = 0.5) -> None:
      self.k = k        # um fator de suavização

      self.tokens: Set[str] = set()
      self.token_depression_counts: Dict[str, int] = defaultdict(int)
      self.token_suicidewatch_counts: Dict[str, int] = defaultdict(int)
      self.depression_messages = self.suicidewatch_messages = 0

   def train(self, messages: Iterable[Message]) -> None:
      for message in messages:
         # Incremente as contagens de mensagens
         if message.suicidewatch:
            self.suicidewatch_messages += 1
         else:
            self.depression_messages += 1

         # Incremente as contagens de palavras
         for token in tokenize(message.text):
            self.tokens.add(token)
            if message.suicidewatch:
               self.token_suicidewatch_counts[token] += 1
            else:
               self.token_depression_counts[token] += 1

   def __probabilities(self, token: str) -> Tuple[float, float]:
      """Retorna P(token | suicidewatch) e P(token | depression)"""
      suicidewatch = self.token_suicidewatch_counts[token]
      depression = self.token_depression_counts[token]

      p_token_suicidewatch = (suicidewatch + self.k) / (self.suicidewatch_messages + 2 * self.k)
      p_token_depression = (depression + self.k) / (self.depression_messages + 2 * self.k)

      return p_token_suicidewatch, p_token_depression
   
   def predict(self, text: str) -> float:
      text_tokens = tokenize(text)
      log_prob_if_suicidewatch = log_prob_if_depression = 0.0

      # Itere em cada palavra do vocabulário
      for token in self.tokens:
         prob_if_suicidewatch, prob_if_depression = self.__probabilities(token)

         # Se o token aparecer na mensagem, adicione o log da probabilidade de vê-lo
         if token in text_tokens:
            log_prob_if_suicidewatch += math.log(prob_if_suicidewatch)
            log_prob_if_depression += math.log(prob_if_depression)
         # Se não, adicione o log da probabilidade de não vê-lo (log(1 - probabilidade de vê-lo))
         else:
            log_prob_if_suicidewatch += math.log(1 - prob_if_suicidewatch)
            log_prob_if_depression += math.log(1 - prob_if_depression)

         prob_if_suicidewatch = math.exp(log_prob_if_suicidewatch)
         prob_if_depression = math.exp(log_prob_if_depression)

         return prob_if_suicidewatch / (prob_if_suicidewatch + prob_if_depression)
      
import random
from typing import TypeVar
X = TypeVar('X')  # tipo genérico para representar um ponto de dados

# Separar parte dos dados para treino e parte para teste
def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
   """Divida os dados em frações [prob, 1 - prob]"""
   data = data[:]                                     # Faça uma cópia superficial
   random.shuffle(data)                               # porque o shuffle modifica a lista.
   cut = int(len(data) * prob)                        # Use prob para encontrar um limiar
   return data[:cut], data[cut:]                      # e dividir a lista aleatória nesse ponto

random.seed(0)                # padronização da seed caso alguém queira replicar o trabalho

data: List[Message] = []

with open('mental-health.csv') as f:
   for line in f:
      depressionwatch = 'depressionwatch' in line[1]
      text = line[0]
      data.append(Message(text, depressionwatch))
      break

train_messages, test_messages = split_data(data, 0.75)

model = NaiveBayesClassifier()
model.train(train_messages)         # treino 