training_data = []


#greetings
training_data.append({"class":"greeting", "sentence":"hi"})
training_data.append({"class":"greeting", "sentence":"hello"})
training_data.append({"class":"greeting", "sentence":"hey"})
training_data.append({"class":"greeting", "sentence":"hola"})
training_data.append({"class":"greeting", "sentence":"howdy"})
training_data.append({"class":"greeting", "sentence":"bonjour"})


#health
training_data.append({"class":"health", "sentence":"How are you Pricy"})
training_data.append({"class":"health", "sentence":"What's up"})
training_data.append({"class":"health", "sentence":"Are you fine"})
training_data.append({"class":"health", "sentence":"How is you health"})
training_data.append({"class":"health", "sentence":"Are you feeling well"})
training_data.append({"class":"health", "sentence":"Are you doing well"})


#work
training_data.append({"class":"work", "sentence":"what can you do for me"})
training_data.append({"class":"work", "sentence":"how do you function"})
training_data.append({"class":"work", "sentence":"what is your work"})
training_data.append({"class":"work", "sentence":"what is your work"})
training_data.append({"class":"work", "sentence":"what do you do"})
training_data.append({"class":"work", "sentence":"what are you trained for"})


#brand1
training_data.append({"class":"brand1", "sentence":"price of samsung"})
training_data.append({"class":"brand1", "sentence":"what is the price of nokia"})
training_data.append({"class":"brand1", "sentence":"what is the cost of mi"})
training_data.append({"class":"brand1", "sentence":"market price of lenevo"})
training_data.append({"class":"brand1", "sentence":"current price of oppo"})
training_data.append({"class":"brand1", "sentence":"compare iphone"})


#brand2
training_data.append({"class":"brand2", "sentence":"price of sony"})
training_data.append({"class":"brand2", "sentence":"what is the price of huawei"})
training_data.append({"class":"brand2", "sentence":"what is the cost of lg"})
training_data.append({"class":"brand2", "sentence":"market price of micromax"})
training_data.append({"class":"brand2", "sentence":"current price of zte"})
training_data.append({"class":"brand2", "sentence":"compare motorola"})


#brand3
training_data.append({"class":"brand3", "sentence":"price of xiaomi"})
training_data.append({"class":"brand3", "sentence":"what is the price of obi"})
training_data.append({"class":"brand3", "sentence":"what is the cost of one plus"})
training_data.append({"class":"brand3", "sentence":"market price of coolpad"})
training_data.append({"class":"brand3", "sentence":"current price of colors"})
training_data.append({"class":"brand3", "sentence":"compare zopo"})


#brand4
training_data.append({"class":"brand4", "sentence":"do you have otto"})
training_data.append({"class":"brand4", "sentence":"what is the price of vivo"})
training_data.append({"class":"brand4", "sentence":"what is the cost of gionee"})
training_data.append({"class":"brand4", "sentence":"market price of htc"})
training_data.append({"class":"brand4", "sentence":"current price of meizu"})
training_data.append({"class":"brand4", "sentence":"compare lava"})


#negative
training_data.append({"class":"negative", "sentence":"No i don't want to buy"})
training_data.append({"class":"negative", "sentence":"No i don't need"})
training_data.append({"class":"negative", "sentence":"i do not want to ask price"})
training_data.append({"class":"negative", "sentence":"i do not need phone"})
training_data.append({"class":"negative", "sentence":"i dont want you to do that"})
training_data.append({"class":"negative", "sentence":"i am not asking that to you"})


#utter
training_data.append({"class":"utter", "sentence":"ok"})
training_data.append({"class":"utter", "sentence":"umm"})
training_data.append({"class":"utter", "sentence":"hmm"})
training_data.append({"class":"utter", "sentence":"huh"})
training_data.append({"class":"utter", "sentence":"oh"})
training_data.append({"class":"utter", "sentence":"good"})


#jokes
training_data.append({"class":"jokes", "sentence":"tell me a joke"})
training_data.append({"class":"jokes", "sentence":"make me laugh"})
training_data.append({"class":"jokes", "sentence":"I am sad."})
training_data.append({"class":"jokes", "sentence":"joke"})
training_data.append({"class":"jokes", "sentence":"make a joke"})
training_data.append({"class":"jokes", "sentence":"can you joke?"})


#laugh
training_data.append({"class":"laugh", "sentence":"haha"})
training_data.append({"class":"laugh", "sentence":"hehe"})
training_data.append({"class":"laugh", "sentence":"lol"})
training_data.append({"class":"laugh", "sentence":"that was funny"})
training_data.append({"class":"laugh", "sentence":"you are funny"})
training_data.append({"class":"laugh", "sentence":"so funny"})


#love
training_data.append({"class":"love", "sentence":"i love you"})
training_data.append({"class":"love", "sentence":"love u"})
training_data.append({"class":"love", "sentence":"i love you so much pricy"})
training_data.append({"class":"love", "sentence":"i like you"})
training_data.append({"class":"love", "sentence":"i lob u"})
training_data.append({"class":"love", "sentence":"i really love you"})


#exit
training_data.append({"class":"exit", "sentence":"ok bye"})
training_data.append({"class":"exit", "sentence":"goodbye"})
training_data.append({"class":"exit", "sentence":"exit"})
training_data.append({"class":"exit", "sentence":"bye bye"})
training_data.append({"class":"exit", "sentence":"talk to you later"})
training_data.append({"class":"exit", "sentence":"see you later"})




print ("%s sentences of training data" % len(training_data))
