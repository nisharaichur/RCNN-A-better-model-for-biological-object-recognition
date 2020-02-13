#dumps the metrics into the json file in the form of dictionary
with open("metrics.json", "w") as fp:
  json.dump(str(metrics), fp)


#load the metrics for plotting
import ast
with open("/content/metrics.json", 'r') as fb:
  met = json.load(fb)

res = ast.literal_eval(met)


#dumps the weights into the txt file
import pickle
with open("weights.txt", 'wb+') as file:
  pickle.dump(net.variables, file)

#loads back the weights 
with open("/content/weights.txt", 'rb') as file:
  var = pickle.load(file)
