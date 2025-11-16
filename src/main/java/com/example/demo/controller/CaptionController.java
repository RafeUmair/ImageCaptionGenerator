package com.example.demo.controller;

import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.io.entity.ByteArrayEntity;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.ContentType;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/api/caption")
@CrossOrigin(origins = "http://localhost:5173") 
public class CaptionController {

    private final String HF_API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning";
    private final String HF_TOKEN = "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"; 

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<String> caption(@RequestParam("image") MultipartFile file) throws IOException {
        try (CloseableHttpClient client = HttpClients.createDefault()) {
            HttpPost post = new HttpPost(HF_API_URL);
            post.setHeader("Authorization", "Bearer " + HF_TOKEN);

            post.setEntity(new ByteArrayEntity(file.getBytes(), ContentType.DEFAULT_BINARY));

            String response = EntityUtils.toString(client.execute(post).getEntity());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Error: " + e.getMessage());
        }
    }
}
