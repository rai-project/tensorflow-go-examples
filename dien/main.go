package main

import (
	"bufio"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	//Parse flags
	modeldir := flag.String("dir", "./", "Directory containing trained model files. Assumes model file is called frozen_inference_graph.pb")
	//datafile := flag.String()
	flag.Parse()
	if *modeldir == "" {
		flag.Usage()
		return
	}

	modelpath := filepath.Join(*modeldir, "DIEN.pb")
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		log.Fatal(err)
	}
	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	source := Dataprocess(16)
	uids, mids, cats, mid_his, cat_his, mid_mask, sl := prepare_data(source)

	// uids, mids, cats, mid_his, cat_his, mid_mask, length_x
	inputOp0 := graph.Operation("Inputs/mid_his_batch_ph")
	inputOp1 := graph.Operation("Inputs/cat_his_batch_ph")
	inputOp2 := graph.Operation("Inputs/uid_batch_ph")
	inputOp3 := graph.Operation("Inputs/mid_batch_ph")
	inputOp4 := graph.Operation("Inputs/cat_batch_ph")
	inputOp5 := graph.Operation("Inputs/mask")
	inputOp6 := graph.Operation("Inputs/seq_len_ph")

	// inputOp7 := graph.Operation("Inputs/noclk_mid_batch_ph")
	// inputOp8 := graph.Operation("Inputs/noclk_cat_batch_ph")
	// inputOp9 := graph.Operation("Inputs/target_ph")

	o1 := graph.Operation("dien/fcn/Softmax")
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputOp0.Output(0): mid_his,
			inputOp1.Output(0): cat_his,
			inputOp2.Output(0): uids,
			inputOp3.Output(0): mids,
			inputOp4.Output(0): cats,
			inputOp5.Output(0): mid_mask,
			inputOp6.Output(0): sl,
			// inputOp7.Output(0): noClkMidHis,
			// inputOp8.Output(0): noClkCatHis,
			// inputOp9.Output(0): target,
		},
		[]tf.Output{
			o1.Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}
	probabilities := output[0].Value().([][]float32)[0]
	fmt.Println(probabilities)

}

func prepare_data(source [][]interface{}) (*tf.Tensor, *tf.Tensor, *tf.Tensor, *tf.Tensor, *tf.Tensor, *tf.Tensor, *tf.Tensor) {
	lengthx := []int32{}
	seqs_mid := [][]int{}
	seqs_cat := [][]int{}

	uidsRaw := []int32{}
	midsRaw := []int32{}
	catsRaw := []int32{}

	for _, v := range source {
		uidsRaw = append(uidsRaw, int32(v[0].(int)))
		midsRaw = append(midsRaw, int32(v[1].(int)))
		catsRaw = append(catsRaw, int32(v[2].(int)))
		lengthx = append(lengthx, int32(len(v[4].([]int))))
		seqs_mid = append(seqs_mid, v[3].([]int))
		seqs_cat = append(seqs_cat, v[4].([]int))
	}

	uids, err := tf.NewTensor(uidsRaw)
	if err != nil {
		log.Fatal(err)

	}
	mids, _ := tf.NewTensor(midsRaw)
	cats, _ := tf.NewTensor(catsRaw)

	n_samples := len(seqs_mid)
	var maxlen_x int32
	for i, e := range lengthx {
		if i == 0 || e >= maxlen_x {
			maxlen_x = e
		}
	}

	mid_his_raw := make([][]int32, n_samples)
	for n := 0; n < n_samples; n++ {
		tn := make([]int32, maxlen_x)
		for m := 0; m < int(maxlen_x); m++ {
			tn[m] = 0
		}
		mid_his_raw[n] = tn
	}

	cat_his_raw := make([][]int32, n_samples)
	for n := 0; n < n_samples; n++ {
		tn := make([]int32, maxlen_x)
		for m := 0; m < int(maxlen_x); m++ {
			tn[m] = 0
		}
		cat_his_raw[n] = tn
	}

	mid_mask_raw := make([][]float32, n_samples)
	for n := 0; n < n_samples; n++ {
		tn := make([]float32, maxlen_x)
		for m := 0; m < int(maxlen_x); m++ {
			tn[m] = 0
		}
		mid_mask_raw[n] = tn
	}

	for idx, _ := range seqs_mid {
		for i := 0; i < int(lengthx[idx]); i++ {
			mid_mask_raw[idx][i] = 1.0
		}
	}

	for idx, sx := range seqs_mid {
		for i := 0; i < int(lengthx[idx]); i++ {
			mid_his_raw[idx][i] = int32(sx[i])
		}
	}

	for idx, sy := range seqs_cat {
		for i := 0; i < int(lengthx[idx]); i++ {
			cat_his_raw[idx][i] = int32(sy[i])
		}
	}

	mid_his, _ := tf.NewTensor(mid_his_raw)
	mid_mask, _ := tf.NewTensor(mid_mask_raw)
	cat_his, _ := tf.NewTensor(cat_his_raw)

	length_x, _ := tf.NewTensor(lengthx)
	return uids, mids, cats, mid_his, cat_his, mid_mask, length_x
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func Dataprocess(batch_size int) [][]interface{} {
	mUid := make(map[string]int)
	mMid := make(map[string]int)
	mCat := make(map[string]int)
	metaMap := make(map[string]string)
	metaIdMap := make(map[int]int)

	fileUid, err := os.Open("uid_voc")
	if err != nil {
		log.Fatal(err)
	}
	defer fileUid.Close()
	scannerUid := bufio.NewScanner(fileUid)

	for scannerUid.Scan() {
		newText := strings.Split(scannerUid.Text(), ",")
		mUid[newText[0]], err = strconv.Atoi(newText[1])
		if err != nil {
			log.Fatal(err)
		}
	}

	fileMid, err := os.Open("mid_voc")
	if err != nil {
		log.Fatal(err)
	}

	defer fileMid.Close()
	scannerMid := bufio.NewScanner(fileMid)

	for scannerMid.Scan() {
		newText := strings.Split(scannerMid.Text(), ",")
		mMid[newText[0]], err = strconv.Atoi(newText[1])
	}

	fileCat, err := os.Open("cat_voc")
	if err != nil {
		log.Fatal(err)
	}
	defer fileCat.Close()

	scannerCat := bufio.NewScanner(fileCat)

	for scannerCat.Scan() {
		newText := strings.Split(scannerCat.Text(), ",")
		mCat[newText[0]], err = strconv.Atoi(newText[1])
	}
	var sourceDicts [3]map[string]int
	sourceDicts[0] = mUid
	sourceDicts[1] = mMid
	sourceDicts[2] = mCat

	fileMeta, err := os.Open("item-info")
	if err != nil {
		log.Fatal(err)

	}
	defer fileMeta.Close()

	scannerMeta := bufio.NewScanner(fileMeta)

	for scannerMeta.Scan() {
		newText := strings.Trim(scannerMeta.Text(), " ")
		newText1 := strings.Split(newText, "\t")
		var ok bool
		_, ok = metaMap[newText1[0]]
		if ok == false {
			metaMap[newText1[0]] = newText1[1]
		}
	}

	for key := range metaMap {
		val := metaMap[key]
		v1, ok1 := sourceDicts[1][key]
		var midIdx int
		if ok1 {
			midIdx = v1
		} else {
			midIdx = 0
		}
		v2, ok2 := sourceDicts[2][val]
		var catIdx int
		if ok2 {
			catIdx = v2
		} else {
			catIdx = 0
		}
		metaIdMap[midIdx] = catIdx
	}

	fileReview, err := os.Open("reviews-info")
	if err != nil {
		log.Fatal(err)
	}
	defer fileReview.Close()

	sourceBuffer := [][]string{}
	file, err := os.Open("local_test_splitByUser")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	cnt := 0
	for scanner.Scan() {
		if cnt == batch_size {
			break
		}
		newText := strings.Trim(scanner.Text(), "\n")
		newText1 := strings.Split(newText, "\t")
		sourceBuffer = append(sourceBuffer, newText1)
		cnt++
	}

	currPath, err := os.Getwd()
	if err != nil {
		panic(err)
	}

	uid_batch, err := os.Create(filepath.Join(currPath, "input/uid_batch.txt"))
	check(err)
	defer uid_batch.Close()
	mid_batch, err := os.Create(filepath.Join(currPath, "input/mid_batch.txt"))
	check(err)
	defer mid_batch.Close()
	cat_batch, err := os.Create(filepath.Join(currPath, "input/cat_batch.txt"))
	check(err)
	defer cat_batch.Close()
	mid_his_batch, err := os.Create(filepath.Join(currPath, "input/mid_his_batch.txt"))
	check(err)
	defer mid_his_batch.Close()
	cat_his_batch, err := os.Create(filepath.Join(currPath, "input/cat_his_batch.txt"))
	check(err)
	defer cat_his_batch.Close()

	//TODO: source Buffer sort by his_length
	var uid int
	var mid int
	var cat int
	source := make([][]interface{}, 0)

	for cnt, ss := range sourceBuffer {
		if cnt == batch_size {
			break
		}
		subsource := make([]interface{}, 7)
		if v, ok := sourceDicts[0][ss[1]]; ok {
			uid = v
		} else {
			uid = 0
		}
		if v, ok := sourceDicts[1][ss[2]]; ok {
			mid = v
		} else {
			mid = 0
		}
		if v, ok := sourceDicts[2][ss[3]]; ok {
			cat = v
		} else {
			cat = 0
		}
		var tmp []int
		textNew := strings.Split(ss[4], "")
		for _, fea := range textNew {
			if v, ok := sourceDicts[1][fea]; ok {
				tmp = append(tmp, v)
			} else {
				tmp = append(tmp, 0)
			}
		}
		midList := tmp

		var tmp1 []int
		textNew = strings.Split(ss[5], "")
		for _, fea := range textNew {
			if v, ok := sourceDicts[2][fea]; ok {
				tmp1 = append(tmp1, v)
			} else {
				tmp1 = append(tmp1, 0)
			}
		}
		catList := tmp1
		subsource[0] = uid
		uid_batch.WriteString(strconv.Itoa(uid) + "\n")

		subsource[1] = mid
		mid_batch.WriteString(strconv.Itoa(mid) + "\n")

		subsource[2] = cat
		cat_batch.WriteString(strconv.Itoa(cat) + "\n")

		subsource[3] = midList
		subsource[4] = catList

		source = append(source, subsource)
	}

	lengthx := []int{}

	for _, v := range source {
		lengthx = append(lengthx, len(v[4].([]int)))
	}

	var maxlen_x int
	for i, e := range lengthx {
		if i == 0 || e > maxlen_x {
			maxlen_x = e
		}
	}

	mask, err := os.Create(filepath.Join(currPath, "input/mask.txt"))
	check(err)
	defer mask.Close()

	seq_len, err := os.Create(filepath.Join(currPath, "input/seq_len.txt"))
	check(err)
	defer seq_len.Close()
	for idx, ss := range source {
		midList := ss[3].([]int)
		catList := ss[4].([]int)

		mid_mask_raw := make([]float32, maxlen_x)
		for m := 0; m < int(maxlen_x); m++ {
			mid_mask_raw[m] = 0
		}

		for i := 0; i < int(lengthx[idx]); i++ {
			mid_mask_raw[i] = 1.0
		}
		mask.WriteString(floatSliceToString(mid_mask_raw) + "\n")

		mid_his_raw := make([]int, maxlen_x)
		for i := 0; i < int(lengthx[idx]); i++ {
			mid_his_raw[i] = midList[i]
		}
		mid_his_batch.WriteString(intSliceToString(mid_his_raw) + "\n")

		cat_his_raw := make([]int, maxlen_x)
		for i := 0; i < int(lengthx[idx]); i++ {
			cat_his_raw[i] = int(catList[i])
		}

		cat_his_batch.WriteString(intSliceToString(cat_his_raw) + "\n")

		seq_len.WriteString(strconv.Itoa(lengthx[idx]) + "\n")
	}

	return source
}

func intSliceToString(in []int) (out string) {
	for ii, v := range in {
		if ii != 0 {
			out += ","
		}
		out += strconv.Itoa(v)
	}
	return
}

func floatSliceToString(in []float32) (out string) {
	for ii, v := range in {
		if ii != 0 {
			out += ","
		}
		out += fmt.Sprintf("%f", v)
	}
	return
}
