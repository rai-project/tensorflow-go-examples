package main

import (
	"bytes"
	"container/heap"
	"fmt"
	"image"
	_ "image/jpeg"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func makeTensorFromImage(filename string) (*tf.Tensor, image.Image, int32, int32, error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, nil, 0, 0, err
	}

	r := bytes.NewReader(b)
	img, _, err := image.Decode(r)
	if err != nil {
		fmt.Println("Input image decoding failed!!!")
		fmt.Println(err)
	}

	x := img.Bounds().Max.X
	y := img.Bounds().Max.Y

	if err != nil {
		return nil, nil, 0, 0, err
	}

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(b))
	if err != nil {
		return nil, nil, 0, 0, err
	}
	// Creates a tensorflow graph to decode the jpeg image
	graph, input, output, newX, newY, err := constructGraphToNormalizeImage(x, y)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	// Execute that graph to decode this one image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		fmt.Println("input image conversion to tensor error", err)
		return nil, nil, 0, 0, err
	}
	return normalized[0], img, newX, newY, nil
}

func constructGraphToNormalizeImage(x, y int) (graph *tf.Graph, input, output tf.Output, H, W int32, err error) {
	inputSize := 346
	W = int32(inputSize)
	// H = int32(1.0 * y * inputSize / x)
	H = int32(346)
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	image := op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))
	image = op.Cast(s, image, tf.Float)
	// NormalizedImage := op.Div(s, jpegImage, op.Max(s, jpegImage, op.Const(s.SubScope("max"), []int32{0, 1, 2})))
	image = op.Div(s, image, op.Const(s, float32(255)))
	output =
		op.ResizeBilinear(s,
			op.ExpandDims(s,
				image,
				op.Const(s.SubScope("make_batch"), int32(0))),
			op.Const(s.SubScope("size"), []int32{H, W}))
	graph, err = s.Finalize()
	return graph, input, output, W, H, err
}

func batchify(tensors []*tf.Tensor) (packedTensor *tf.Tensor) {
	s := op.NewScope()
	var pack []tf.Output
	for i := 0; i < len(tensors); i++ {
		input := op.Placeholder(s.SubScope("input"), tf.Float)
		pack = append(pack, input)
	}
	output := op.Pack(s, pack)

	graph, err := s.Finalize()
	if err != nil {
		panic(err.Error())
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		panic(err.Error())
	}
	defer session.Close()

	inputFeed := make(map[tf.Output]*tf.Tensor)
	for i := 0; i < len(tensors); i++ {
		inputFeed[pack[i]] = tensors[i]
	}
	batchOutput, err := session.Run(
		inputFeed,
		[]tf.Output{output},
		nil)
	if err != nil {
		fmt.Println("input image conversion to tensor error", err)
	}

	return batchOutput[0]
}

type argSort struct {
	Args []float32
	Idxs []int64
}

// Implement sort.Interface Len
func (s argSort) Len() int { return len(s.Args) }

// Implement sort.Interface Less
func (s argSort) Less(i, j int) bool { return s.Args[i] < s.Args[j] }

// Implment sort.Interface Swap
func (s argSort) Swap(i, j int) {
	// swap value
	s.Args[i], s.Args[j] = s.Args[j], s.Args[i]
	// swap index
	s.Idxs[i], s.Idxs[j] = s.Idxs[j], s.Idxs[i]
}

func main() {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "2")
	// Parse flags
	modeldir := "./"
	jpgfile := "./COCO_val2014_000000224477.jpg"
	// outjpg := "output.jpg"

	modelpath := filepath.Join(modeldir, "im2txt_frozen_graph.pb")
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		fmt.Println(err)
		log.Fatal(err)
	}

	// Construct the vocabulary
	// vocab, reverseVocab := constructVocabulary("word_counts_p.txt")

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		fmt.Println(err)
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		fmt.Println("create new session error", err)
		log.Fatal(err)
	}
	defer session.Close()

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, _, _, _, err := makeTensorFromImage(jpgfile)
	if err != nil {
		fmt.Println("makeTensorFromImage error", err)
		log.Fatal(err)
	}

	// Get all the input and output operations
	inputOp := graph.Operation("control_dependency")
	outputOp := graph.Operation("lstm/initial_state")

	// Execute COCO Graph
	initialState, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputOp.Output(0): tensor,
		},
		[]tf.Output{
			outputOp.Output(0),
		},
		nil)
	if err != nil {
		fmt.Println("session run error", err)
		log.Fatal(err)
	}

	// ptInit := initialState[0].Value().([][]float32)
	// for i, val := range ptInit[0] {
	// 	fmt.Printf("%e ", val)
	// 	if i%4 == 0 && i != 0 {
	// 		fmt.Printf("\n")
	// 	}
	// }

	// Beam Search
	beamSize := 3
	maxCaptionLength := 20
	lengthNormalizationFactor := 0.0

	vocabFilename := "./word_counts_p.txt"
	vocab := constructVocabulary(vocabFilename)

	initialStateArray, _ := tf.NewTensor(initialState[0].Value().([][]float32)[0])
	initialBeam := caption{
		sentence: []int64{vocab.startID},
		state:    initialStateArray,
		logprob:  0.0,
		score:    0.0}

	partialCaption := &topN{n: beamSize}
	heap.Init(partialCaption)
	partialCaption.PushTopN(initialBeam)

	completeCaption := &topN{n: beamSize}
	heap.Init(completeCaption)

	intermediateInputFeed := graph.Operation("input_feed")
	intermediateStateFeed := graph.Operation("lstm/state_feed")
	softmaxOp := graph.Operation("softmax")
	stateOp := graph.Operation("lstm/state")

	for i := 0; i < maxCaptionLength-1; i++ {
		partialCaptionList := partialCaption.Extract()
		partialCaption.Reset()

		inputFeed := []int64{}
		stateFeed := []*tf.Tensor{}
		for _, cap := range partialCaptionList {
			inputFeed = append(inputFeed, cap.sentence[len(cap.sentence)-1])
			// fmt.Println(cap.state.Shape())
			stateFeed = append(stateFeed, cap.state)
		}
		fmt.Println("inputFeed:", inputFeed)
		inputTensor, err := tf.NewTensor(inputFeed)
		if err != nil {
			fmt.Println("inputTensor error:", err)
		}
		stateTensor := batchify(stateFeed)

		// fmt.Println("stateTensor shape:", stateTensor)
		// fmt.Printf("inputTensor shape: %v\n", inputTensor.Shape())
		// fmt.Printf("stateTensor shape: %v\n", stateTensor.Shape())

		intermediateOutput, err := session.Run(
			map[tf.Output]*tf.Tensor{
				intermediateInputFeed.Output(0): inputTensor,
				intermediateStateFeed.Output(0): stateTensor,
			},
			[]tf.Output{
				softmaxOp.Output(0),
				stateOp.Output(0),
			},
			nil)
		if err != nil {
			fmt.Println("intermediate session run error:", err)
			log.Fatal(err)
		}

		softmaxOutput := intermediateOutput[0].Value().([][]float32)
		stateOutput := intermediateOutput[1].Value().([][]float32)

		for j, cap := range partialCaptionList {
			wordProbabilities := softmaxOutput[j]
			state := stateOutput[j]
			fmt.Println(wordProbabilities[0:100])
			// fmt.Println(state)
			break

			wordLen := len(wordProbabilities)
			idxs := []int64{}
			for idx := int64(0); idx < int64(wordLen); idx++ {
				idxs = append(idxs, idx)
			}
			arg := argSort{Args: wordProbabilities, Idxs: idxs}
			sort.Sort(arg)
			mostLikelyWords := arg.Idxs[wordLen-beamSize : wordLen]
			fmt.Println(wordProbabilities[wordLen-beamSize : wordLen])
			fmt.Println(mostLikelyWords)

			for k := len(mostLikelyWords) - 1; k >= 0; k-- {
				w := mostLikelyWords[k]
				p := wordProbabilities[w]
				if p < 1e-12 {
					continue
				}

				sentence := append(cap.sentence, w)
				logprob := cap.logprob + float32(math.Log(float64(p)))
				score := logprob
				stateTensor, _ := tf.NewTensor(state)
				// fmt.Println("inner stateTensor shape:", stateTensor.Shape())

				if w == vocab.endID {
					if lengthNormalizationFactor > 0 {
						score /= float32(math.Pow(float64(len(sentence)), lengthNormalizationFactor))
					}

					beam := caption{
						sentence: sentence,
						state:    stateTensor,
						logprob:  logprob,
						score:    score}
					completeCaption.PushTopN(beam)
				} else {
					beam := caption{
						sentence: sentence,
						state:    stateTensor,
						logprob:  logprob,
						score:    score}
					partialCaption.PushTopN(beam)
				}
			}
		}
		if partialCaption.Len() == 0 {
			break
		}
	}

	if completeCaption.Len() == 0 {
		completeCaption = partialCaption
	}

	captions := completeCaption.Extract()
	wordArray := []string{}
	IDArray := []int64{}
	for i, caption := range captions {
		//print the final result
		for _, wordID := range caption.sentence {
			wordArray = append(wordArray, vocab.reverseVocab[wordID])
			IDArray = append(IDArray, wordID)
		}
		predSentence := strings.Join(wordArray[1:len(wordArray)-1], " ")
		fmt.Printf("%d) %s (p=%f)   ", i, predSentence, math.Exp(float64(caption.logprob)))
		fmt.Println(IDArray)
		wordArray = nil
		IDArray = nil
	}
}
