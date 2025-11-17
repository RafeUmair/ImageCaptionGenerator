package com.example.demo.controller;

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.example.demo.service.CaptionService;

@RestController
@RequestMapping("/api/caption")
@CrossOrigin(origins = "http://localhost:5173")
public class CaptionController {

    private final CaptionService captionService;

    public CaptionController(CaptionService captionService) {
        this.captionService = captionService;
    }

   @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
public ResponseEntity<String> caption(@RequestParam("image") MultipartFile file) throws Exception {
    String caption = captionService.generateCaption(file.getBytes());
    return ResponseEntity.ok(caption);
}
}
