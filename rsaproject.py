import socket
import math

class rsa:
    def __init__(self, send_or_receive):
        self.send_or_receive=send_or_receive


        def isPrime(self,n):
            for i in range(2,int(n**0.5)+1):
                if n%i==0:
                    return False
            return True

        def find_inverse(self,remainder,modulus):
            return pow(remainder,-1,modulus)

        def translate_to_numbers(self,input):
            output=''
            for i in input:
                output+=str((ord(i)))
            return output

        def translate_to_letters(self,input):
            output=str('')
            input=str(input)
            for i in range(0,len(input),2):
                output+=chr(int(input[i]+input[i+1]))
            return output

        def encrypt(self,my_message,my_public_key):
            self.w=translate_to_numbers(self,self.my_message)
            print(self.w)
            self.c=pow(w,my_public_key[1],my_public_key[0])
            print(self.c)
            return self.c

        def decrypt(self,my_cipher,my_private_key):
            self.w_prime=pow(my_cipher,my_private_key)
            decrypted_message=translate_to_numbers(self,w_prime)
            print(decrypted_message)
            return decrypted_message


        if(self.send_or_receive=="r"):
            self.p=int(input("enter a prime: "))
            self.q=int(input("enter a second prime to form \"m\": "))
            if ((isPrime(self,self.p)==False) or (isPrime(self,self.q)==False)):
                print("invalid. p and q must be prime")
            self.m=self.p*self.q
            self.phim=(self.p-1)*(self.q-1)
            print("m="+str(self.m))
            self.e=int(input("choose encrypting exponent \"e\" coprime to Φ(m)="+str(self.phim)+": "))
            self.public_key=[self.m,self.e]
            self.d=find_inverse(self,self.e,self.phim)
            print("decrypting exponent \"d\" is",self.d)
            self.ready=input("type send to send public key ")
            if(self.ready=="send"):
                conn.sendall(str(self.public_key).encode())


        elif(send_or_receive=="s"):
            self.my_message=input("what message would you like to send? ")
            self.c=encrypt(self,self.my_message,self.my_public_key)
            self.ready=input("type send to send message ")
            if(self.ready=="send"):
                conn.sendall((self.c).encode())

        else:
            print("invalid answer")




HOST = "169.231.217.208"  # Standard loopback interface address (localhost)
PORT = 65441  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        print("Waiting to receive messages...") 
        while True:
            data = conn.recv(1024)
            data=data.decode()
            if(data =='exit'):
                break
            print(("Received message: ") , data)
            message_to_send=input("Send message: ")
            if(message_to_send[0]=="/"):
                exec(message_to_send[1:])
            conn.sendall((message_to_send).encode())
            if(message_to_send =='exit'):
                break


