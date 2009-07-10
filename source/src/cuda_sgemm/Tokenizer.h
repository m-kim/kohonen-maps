#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <vector>
#include <string>
using namespace std;

class Tokenizer
{
public:
	Tokenizer(string &str,
			const string &delimiter = " "){
		string::size_type lastPos = str.find_first_not_of(delimiter, 0);
		string::size_type pos = str.find_first_of(delimiter, lastPos);
		while (string::npos != pos || string::npos != lastPos)
		    {
		        // Found a token, add it to the vector.
		        tokens.push_back(str.substr(lastPos, pos - lastPos));
		        // Skip delimiters.  Note the "not_of"
		        lastPos = str.find_first_not_of(delimiter, pos);
		        // Find next "non-delimiter"
		        pos = str.find_first_of(delimiter, lastPos);
		    }

	}

	vector<string> tokens;

	string operator()(int index){
		return tokens[index];
	}

};
#endif
