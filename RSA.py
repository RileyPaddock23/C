import math
#public key (m,e) where m=pq for p and q distinct primes. 1 less than equals e less than equals (p-1)(q-1)-phi(pq) with gcd(e,phi(pq))=1
#private key d integer with dbar times ebar = 1bar in z/phi(pq)z
#step 1 translate message into seuquence of digits to get number w
#step 2 w^e = c(modm) and send c
# c^d (mod m) to decrypt


class rsa:


	def __init__(self, send_or_receive):
		self.send_or_receive=send_or_receive
		self.letter_to_number_key={' ':'00','a':'01','b':'02','c':'03','d':'04','e':'05','f':'06','g':'07','h':'08','i':'09','j':'10','k':'11','l':'12','m':'13','n':'14','o':'15','p':'16','q':'17','r':'18','s':'19','t':'20','u':'21','v':'22','w':'23','x':'24','y':'25','z':'26'}
		self.number_to_letter_key = dict((v,k) for k,v in self.letter_to_number_key.items())




		def phi(self,n):
			sum=0
			for i in range(1,n):
				if math.gcd(i,n)==1:
					sum+=1
			return sum

		def find_inverse(self,remainder,modulus):
			return pow (remainder, -1, modulus)

		def translate_to_numbers(self,input):
			output=''
			for i in str(input):
				output+=str(self.letter_to_number_key[i])
			return (output)

		def translate_to_letters(self,input):
			output=str('')
			for i in range(0,len(input),2):
				output+=str(self.number_to_letter_key[(input[i]+input[i+1])])
			return output

		def encrypt(self,message):
			w=int(self.translate_to_numbers(message))
			c=pow(w,self.public_key[1],self.public_key[0])
			return c

		def decrypt(self,c, private_key, public_key):
			original_message=pow(c,private_key,public_key[0])
			if len(str(original_message))%2==1:
					return ('0'+str(original_message))
			return translate_to_letters((str(original_message)))



		if(self.send_or_receive=="receive"):
			self.p=int(input("enter a prime: "))
			self.q=int(input("enter a second prime to form \"m\": "))
			self.m=self.p*self.q
			self.phim=phi(self,self.m)
			print("m="+str(self.m))
			self.e=int(input("choose encrypting exponent \"e\" coprime to Φ(m)="+str(self.phim)+": "))
			self.public_key=[self.m,self.e]
			self.d=find_inverse(self,self.e,self.phim)
			print("decrypting exponent \"d\" is",self.d)
			self.ready=input("type send to send public key")

		elif(send_or_receive=="send"):
			self.message=input("what message would you like to send?")
			self.w=translate_to_numbers(self,self.message)
			print("sending",self.w)

		else:
			print("invalid answer")

	
	


