#ifndef _H_HTTPCL_H_
#define _H_HTTPCL_H_

#include <string>

#define BUFFER_SIZE 4096
#define HTTP_REQ "/pzero"

using namespace std;

class http{
	public:
		http(string host, string port = "80"){
			this->port = port; 
			this->host = host; 
			init(); 
		} 
		int init();
		void addHeader(string, string);
		void finalize(); 
		string get(string);
		void start(); 
		void readResp();
		
	private: 
		int sock; 
		bool connected = false; 
		string port; 
		string msg; 
		string host; 
		string rcv; 
};

#endif

