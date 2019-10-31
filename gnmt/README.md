

Add the following code to inference.py

```
  out_dir = "/home/abduld/mlperf/inference/v0.5/translation/gnmt/tensorflow/savedmodel"
  # Create savedmodel
  with  tf.Session(graph=infer_model.graph) as sess:
    loaded_model, global_step = model_helper.create_or_load_model(
        infer_model.model, out_dir, sess, "infer_name")
    # tf.saved_model.save(loaded_model, out_dir)
    # ckpt_path = loaded_model.saver.save(
    #     sess, os.path.join(out_dir, "translate.ckpt"),
    #     global_step=global_step)

    # Export checkpoint to SavedModel
    builder = tf.saved_model.builder.SavedModelBuilder(out_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.TRAINING, tf.saved_model.tag_constants.SERVING],
                                         strip_default_attrs=True)
    builder.save()
  exit
```