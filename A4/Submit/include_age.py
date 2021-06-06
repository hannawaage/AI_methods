import numpy as np
import pandas as pd
import inspect


class Attribute:
  def __init__(self, name, values, continous=False):
    self.name = name
    self.values = values
    self.continous = continous


class Node:
  def __init__(self, attribute):
    self.attribute = attribute
    self.children = {}

  def add_child(self, branch_val, value):
    self.children[branch_val] = value

  def print_tree(self):
    print(self.attribute.name)
    print("\n")
    for key, value in self.children.items():
      if isinstance(value, Node):
        print("\n")
        print("branch: ", key, ", from ", self.attribute.name)
        value.print_tree()
      else:
        print("branch:", key, "val:", value)

  def test_model(self, test_set):
    classified = []
    for ind, row in test_set.iterrows():
      chosen_branch = test_set[self.attribute.name][ind]
      current_node = self.children[chosen_branch]
      while True:
        if current_node.attribute.continous:
          attr = current_node.attribute
          val = test_set[attr.name][ind]
          branch_val = float(list(current_node.children.keys())[0])
          if val < branch_val:
            chosen_branch = branch_val
          else:
            chosen_branch = " >= " + str(round(branch_val, 3))
        else:
          chosen_branch = test_set[current_node.attribute.name][ind]
        next_node = current_node.children[chosen_branch]
        if not isinstance(next_node, Node):
          classified.append(next_node)
          break
        current_node = next_node
    return classified


def mode(examples):
  return examples.mode().values[0]


def same_class(examples):
  # compare first value to rest. Returns true if all values are identical
  examples_np = examples.to_numpy()
  return (examples_np[0] == examples_np).all()


def entropy(prob):
  if 0 < prob < 1:
    return -(prob*np.log2(prob) + (1 - prob)*np.log2(1 - prob))
  else:
    return 0


def remainder(attribute, examples, class_name):
  p_ks = []
  n_ks = []
  if attribute.continous:
    E_less = examples[examples[attribute_name] < attribute.values]
    if not E_less.empty:
        p_ks.append(len(E_less[E_less[class_name] == 1]))
        n_ks.append(len(E_less[E_less[class_name] == 0]))
    E_above = examples[examples[attribute_name] >= attribute.values]
    if not E_above.empty:
        p_ks.append(len(E_above[E_above[class_name] == 1]))
        n_ks.append(len(E_above[E_above[class_name] == 0]))
  else:
    for value in attribute.values:
      E_k = examples[examples[attribute.name] == value]
      if not E_k.empty:
        p_ks.append(len(E_k[E_k[class_name] == 1]))
        n_ks.append(len(E_k[E_k[class_name] == 0]))
  sum = 0
  for i in range(len(p_ks)):
    prob = float(p_ks[i])/(p_ks[i] + n_ks[i])
    B = entropy(prob)
    sum += (p_ks[i] + n_ks[i])*B
  return sum*(1/len(examples))


def gain(attribute, examples, class_name):
  temp = examples[attribute_name]
  p = len(temp[examples["Survived"] == 1])
  n = len(temp[examples["Survived"] == 0])
  prob = p/(p+n)
  return entropy(prob) - remainder(attribute, examples, class_name)


def choose_attribute(attributes, examples, class_name):
  gains = []
  for A in attributes:
    gains.append(gain(A, examples, class_name))
  gains = np.array(gains)
  return attributes[np.argmax(gains)]


def DTL(examples, attributes, default, class_name):

  update_split = True

  if examples.empty:
    return default
  elif same_class(examples[class_name]):
    return mode(examples[class_name])
  elif not len(attributes):
    return mode(examples[class_name])
  else:
    best = choose_attribute(attributes, examples, class_name)
    rest = attributes.copy()
    rest.remove(best)
    tree = Node(best)
    if best.continous:
      # Update mean
      if update_split:
        best.values = get_cont_split(examples[best.name])
      branch = round(best.values, 3)
      e_xi = examples[examples[best.name] < best.values]
      subtree = DTL(e_xi, rest, mode(examples[class_name]), class_name)
      tree.add_child(branch, subtree)
      branch = " >= " + str(round(best.values, 3))
      e_xi = examples[examples[best.name] >= best.values]
      subtree = DTL(e_xi, rest, mode(examples[class_name]), class_name)
      tree.add_child(branch, subtree)
    else:
      for value in best.values:
        e_xi = examples[examples[best.name] == value]
        subtree = DTL(e_xi, rest, mode(examples[class_name]), class_name)
        tree.add_child(value, subtree)
  return tree


def calculate_accuracy(classified, labels):
  classified = np.array(classified)
  labels = np.array(labels)
  correct = (classified == labels).sum().item()
  return correct/len(classified)


def get_cont_split(examples):
  return examples.mean()


if __name__ == '__main__':
  # Load data
  rawtrain = pd.read_csv("train.csv")
  rawtest = pd.read_csv("test.csv")

  train = rawtrain[["Survived", "Pclass", "Sex", "Age"]]
  test = rawtest[["Survived", "Pclass", "Sex", "Age"]]  # , "Parch", "SibSp"]]

  train["Age"].fillna(train["Age"].mean())
  test["Age"].fillna(test["Age"].mean())

  # Get attribute values
  class_name = "Survived"
  attributes = []
  for attribute_name in train.keys():
    if attribute_name == class_name:
      continue
    values = []
    examples = train[attribute_name].sort_values()
    while not examples.empty:
      value = examples.values[0]
      if isinstance(value, float):
        # Attribute is continous
        values = get_cont_split(examples)
        A = Attribute(attribute_name, values, continous=True)
        break
      else:
        values.append(value)
        examples = examples[examples != value]
        A = Attribute(attribute_name, values)
    attributes.append(A)
  default = mode(train[class_name])

  # Construct tree
  tree = DTL(train, attributes, default, class_name)
  tree.print_tree()
