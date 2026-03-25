import numpy as np
import tensorflow as tf
import cv2
import os


# ---------------- GRADCAM HEATMAP ----------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())

    heatmap /= max_val
    return heatmap.numpy()


# ---------------- IMPROVED OVERLAY ----------------
def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)

    # ---------- STEP 1: CREATE BETTER LEAF MASK ----------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Wider and more accurate green range
    lower_green = np.array([20, 30, 30])
    upper_green = np.array([95, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_green, upper_green)

    # ---------- STEP 2: CLEAN MASK ----------
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

    # Keep only largest green region (main leaf)
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(leaf_mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
        leaf_mask = clean_mask

    # ---------- STEP 3: PREPARE HEATMAP ----------
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # ---------- STEP 4: APPLY MASK TO COLORED HEATMAP ----------
    heatmap_masked = cv2.bitwise_and(heatmap_color, heatmap_color, mask=leaf_mask)

    # ---------- STEP 5: OVERLAY ----------
    superimposed = cv2.addWeighted(heatmap_masked, alpha, img, 1 - alpha, 0)

    # ---------- SAVE ----------
    out_path = os.path.join("static/uploads", "gradcam_" + os.path.basename(img_path))
    cv2.imwrite(out_path, superimposed)

    return out_path
