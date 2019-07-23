import keras.backend as K


anchors = [(1, 1), (2, 2), (3, 3)]
a = [[1, 2, 3], [4, 5, 6]]
b = K.get_session().run(K.reshape(a, [2, 3]))
print(b)
anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, 3, 2])
print(K.constant(anchors))
