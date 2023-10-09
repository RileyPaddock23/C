# echo-client.py

import socket

# from RSA import rsa

class rsa:
    def __init__(self, send_or_receive, s = None, public_key = []):
        self.send_or_receive=send_or_receive
        self.s = s
        self.my_public_key = public_key

        if(self.send_or_receive=="r"):
            self.p=int(input("enter a prime: "))
            self.q=int(input("enter a second prime to form \"m\": "))
            if ((self.isPrime(self.p)==False) or (self.isPrime(self.q)==False)):
                print("invalid. p and q must be prime")
            self.m=self.p*self.q
            self.phim=(self.p-1)*(self.q-1)
            print("m="+str(self.m))
            self.e=int(input("choose encrypting exponent \"e\" coprime to Φ(m)="+str(self.phim)+": "))
            self.public_key=[self.m,self.e]
            self.d=self.find_inverse(self.e,self.phim)
            print("decrypting exponent \"d\" is",self.d)
            self.ready=input("type send to send public key ")
            if(self.ready=="send"):
                self.s.sendall(str(self.public_key).encode())


        elif(send_or_receive=="s"):
            my_message=input("what message would you like to send? ")
            message = []
            for char in my_message:
                c = self.encrypt(char,self.my_public_key)
                message.append(c)
            self.ready=input("type send to send message ")
            if(self.ready=="send"):
                self.s.sendall(("*"+str(message)).encode())

        else:
            print("invalid answer")

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
        return int(output)

    def translate_to_letters(self,input):
        output=str('')
        input=str(input)
        for i in range(0,len(input),2):
            output+=chr(int(input[i]+input[i+1]))
        return output

    def encrypt(self,my_message,my_public_key):
        w=self.translate_to_numbers(my_message)
        print(w)
        c=pow(w,my_public_key[1],my_public_key[0])
        print(c)
        return c

    def decrypt(self,my_cipher,my_private_key):
        w_prime=pow(my_cipher,my_private_key)
        decrypted_message=self.translate_to_numbers(w_prime)
        print(decrypted_message)
        return decrypted_message


        



HOST = "169.231.217.208"  # The server's hostname or IP address
PORT = 65441  # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b"Hello, David")
    e = 10
    d = 5
    m = 20
    while True:
        data = s.recv(1024)
        if f"Received {data.decode()!r}" == "exit":
            break
        
        rec = data.decode()
        if rec[0] == "[":
            split = rec.index(",")
           
            m = int(rec[1:split])
            e = int(rec[split+1:-1])
            print(f"Recieved {[m,e]}")
            x = rsa('s', s, [m,e])
            
        
        else:
            try: 
                value = int(rec)
                decrypt = str(pow(value, d, m))
                text = [chr(int(decrypt[2*i:2*i+2])+87) for i in range(len(str(decrypt))//2)]
                print(f"Recieved {''.join(text)}")
            except ValueError:
                print(f"Received {rec}")

        if data:
            # m = rsa("send")
            msg = input("Message: ")
            if msg == "exit":
                    break

            if msg[0:2] == "C:":
                num = ""
                for char in msg[2:]:
                    if char == " ":
                        num += "37"
                    else:
                        num += str(int(ord(char)-87))

                code = pow(int(num), e, m)
                s.sendall(str(code).encode())
            else:
                s.sendall((msg).encode())






