//#include "pch.h"
#include "httpcl.hpp"
#include <iostream>
#include <string>
#include <cstring>
#include <unistd.h> 
#include <sys/socket.h>
#include <arpa/inet.h>

//#define endm msg+="\r\n"
using namespace std;

int http::init(){
	struct sockaddr_in serv_addr; 
	if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) { 
		perror("Socket creation error"); 
		return -1; 
	} 
	   
	serv_addr.sin_family = AF_INET; 
	serv_addr.sin_port = htons(stoi(port)); 
	   
	// Convert IPv4 and IPv6 addresses from text to binary form 
	if(inet_pton(AF_INET, host.c_str(), &serv_addr.sin_addr)<=0)  
	{ 
		perror("Invalid address | Address not supported "); 
		return -1; 
	} 
   
	if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) 
	{ 
		perror("Connection Failed "); 
		return -1; 
	} 
	connected = true; 
	return 0; 
}
		
void http::addHeader(string name, string val){
	if(connected == false)
		return; 
	msg += name;
	msg += ": ";
	msg += val; 
	msg += "\r\n"; 
}

void http::finalize(){
	msg += "\r\n"; 
}
		
string http::get(string url){
	if(connected == false)
		return ""; 
	msg = "GET " + url + " HTTP/1.1\r\n" + msg; 
	addHeader("Connection", "keep-alive");
	finalize(); 
	
	send(sock, msg.c_str(), msg.length(), 0);
	readResp();
	return this->rcv; 
}

void http::start(){
	if(connected == false)
		return; 
	rcv.clear(); 
	msg.clear(); 
} 
		
void http::readResp(){
	int rcvl, c_size = BUFFER_SIZE, c_cursor = 0; 
	//string rcvs;
	char *buf = (char *)malloc(c_size);
			
	while((rcvl = recv(sock, (buf+c_cursor), c_size-c_cursor, 0)) > 0){
		c_cursor+=rcvl; 
		if(c_cursor == c_size){ // buffer filled, reallocate 
			c_size*=2;
			buf = (char *)realloc(buf,c_size); 
		}else
			break; // escape loop 
		rcvl = 0; 
	} 
	rcv = buf; 
}




