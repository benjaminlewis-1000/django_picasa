#! /usr/bin/env python

import hashlib
import binascii
import numpy as np
import cv2
from textwrap import wrap # for splitting string
import os
import shutil

str1_pre = '4dc968ff0ee35c209572d4777b721587d36fa7b21bdc56b74a3dc0783e7b9518afbfa' + \
	'200a8284bf36e8e4b55b35f427593d849676da0d1555d8360fb5f07fea2'
str2_pre = '4dc968ff0ee35c209572d4777b721587d36fa7b21bdc56b74a3dc0783e7b9518afbfa' + \
	'202a8284bf36e8e4b55b35f427593d849676da0d1d55d8360fb5f07fea2'
print(len(str1_pre))

mutual_len = 12000 - len(str1_pre)
print(mutual_len)
mutual = str(binascii.b2a_hex(os.urandom(int(mutual_len // 2))))
mutual = mutual[2:-1]
print(len(mutual))

# mutual = 'c6b384c4968b28812b676b49d40c09f8af4ed4cc0e306561559aa787d00bc6f70bbdfe3404cf03659e744f8534c00ffb659c4c8740cc942feb2da115a3f415dcbb8607497386656d7d1f34a42059d78f5a8dd1ef0e306561559aa787d00bc6f70bbdfe3404cf03659e704f8534c00ffb659c4c8740cc942feb2da115a3f4155cbb8607497386656d7d1f34a42059d78f5a8dd1efc728d8d93091e9c7b87b43d9e33829379231d7ca' + 'a1' * 5768

# Combined, this gives us a string of length 50 * 80 * 3.

str1 = str1_pre + mutual
str2 = str2_pre + mutual

# print(str1)

hex1 = binascii.unhexlify(str1)
hex2 = binascii.unhexlify(str2)

print(hex1[0:5])

# print((hex1).encode('utf-8'))

m = hashlib.md5()
m.update(hex1) # .encode('utf-8'))
dig_1 = m.hexdigest()

split1 = wrap(str1, 2)
j = hashlib.md5()
for a in split1:
	h = binascii.unhexlify(a)
	j.update(h)

print([binascii.unhexlify(h) for h in split1[:5]])
print("Per: ",  j.hexdigest())

n = hashlib.md5()
n.update(hex2) #.encode('utf-8'))
dig_2 = n.hexdigest()

print(dig_1)
print(dig_2)
assert dig_1 == dig_2
print(len(str1))

# print('desired hash is 008ee33a9d58b51cfeb425b0959121c9')

print("String 1 and two are the same? ", str1 == str2)
print("But the hashes are the same? ", dig_1 == dig_2)

def str_to_numpy(string):
	assert len(string) == 12000
	spl = [int(x, 16) for x in wrap(string, 2)]
	arr = np.array(spl, dtype=np.uint8)
	arr = arr.reshape(50, 40, 3)
	return arr

def hash_numpy_array(np_array):
	m = hashlib.md5()
	arr = np_array.reshape(-1)
	for idx in range(0,len(arr),len(arr)//1000):
		it = arr[idx]
		m.update(bytes([it]))
	# for it in np.nditer(arr[]):
	return m.hexdigest()

def hash_img(img_path):
	pixels = cv2.imread(img_path)
	hashval = hash_numpy_array(pixels)
	return hashval

array1 = str_to_numpy(str1)
array2 = str_to_numpy(str2)

print(hash_numpy_array(array1))
print(hash_numpy_array(array2))

import imageio
imageio.imsave('outfile1.png', array1[:, :, (2, 1, 0)])
shutil.copy2('outfile1.png', 'outfile1.jpg')
imageio.imsave('outfile2.png', array2[:, :, (2, 1, 0)])
shutil.copy2('outfile2.png', 'outfile2.jpg')

print(hash_img('outfile1.jpg'))
print(hash_img('outfile2.jpg'))

# bb = cv2.imread('outfile1.jpg')
# jj = bb - array1
# print(np.mean(jj))
# cc = cv2.imread('outfile2.jpg')
# jj = cc - array2
# print(np.mean(jj))
# print(bb[0:5][0][3])
# print(array1[0:5][0][3])

# print(np.mean(bb - cc))
# assert not np.all(bb == cc)