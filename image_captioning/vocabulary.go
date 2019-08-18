package main

import (
	"bufio"
	"log"
	"os"
	"strconv"
	"strings"
)

type vocabularyT struct {
	vocab        map[string]int
	reverseVocab map[int]string
	startID      int
	endID        int
	unknownID    int
}

func constructVocabulary(vocabFile string) (vocabulary vocabularyT) {
	vocab := make(map[string]int)
	reverseVocab := make(map[int]string)
	startWord := "<S>"
	endWord := "</S>"
	unknownWord := "<UNK>"

	file, err := os.Open(vocabFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		// fmt.Println(scanner.Text())
		wordPair := strings.Split(scanner.Text(), " ")
		word := wordPair[0]
		id, _ := strconv.Atoi(wordPair[1])
		vocab[word] = id
		reverseVocab[id] = word
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	startID := vocab[startWord]
	endID := vocab[endWord]
	unknownID := vocab[unknownWord]

	vocabulary = vocabularyT{
		vocab:        vocab,
		reverseVocab: reverseVocab,
		startID:      startID,
		endID:        endID,
		unknownID:    unknownID}

	return vocabulary
}

// func main() {
// 	vocab := make(map[string]int)
// 	reverseVocab := make(map[int]string)
// 	file, err := os.Open("word_counts_p.txt")
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	defer file.Close()

// 	scanner := bufio.NewScanner(file)
// 	for scanner.Scan() {
// 		// fmt.Println(scanner.Text())
// 		wordPair := strings.Split(scanner.Text(), " ")
// 		word := wordPair[0]
// 		id, _ := strconv.Atoi(wordPair[1])
// 		vocab[word] = id
// 		reverseVocab[id] = word
// 	}

// 	if err := scanner.Err(); err != nil {
// 		log.Fatal(err)
// 	}
// }
