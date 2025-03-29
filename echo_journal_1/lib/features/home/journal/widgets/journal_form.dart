import 'package:flutter/material.dart';
import 'package:echo_journal_1/utils/toast_helper.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

class JournalFormWidget extends StatefulWidget {
  final Function(Map<String, dynamic>) onSubmit;
  final Function(Map<String, dynamic>, String) onUpdate;
  final bool isDarkMode;

  const JournalFormWidget({
    super.key,
    required this.onSubmit,
    required this.onUpdate,
    required this.isDarkMode,
  });

  @override
  JournalFormWidgetState createState() => JournalFormWidgetState();
}

class JournalFormWidgetState extends State<JournalFormWidget> {
  final titleController = TextEditingController();
  final contentController = TextEditingController();
  String? editingJournalId;
  final FlutterTts flutterTts = FlutterTts();
  final stt.SpeechToText _speech = stt.SpeechToText();
  bool _isListening = false;
  String _recognizedText = '';
  String _selectedLanguage = 'en';
  bool _isNepaliMode = false;

  // A set of Nepali Unicode character ranges
  static final _nepaliRange = RegExp(r'[\u0900-\u097F]');
  // A set of English character ranges
  static final _englishRange = RegExp(r'[a-zA-Z]');

  @override
  void initState() {
    super.initState();
    _initializeSpeech();
    // Add listener to content controller for language detection
    contentController.addListener(_detectLanguage);
  }

  // Method to detect language based on text content
  void _detectLanguage() {
    if (!mounted) return;
    final text = contentController.text;
    if (text.isEmpty) return;

    // Count characters in each language range
    int nepaliCount = _countMatches(text, _nepaliRange);
    int englishCount = _countMatches(text, _englishRange);

    // Only change language mode if there's a clear preference and it differs from current setting
    if (nepaliCount > 0 && englishCount == 0 && !_isNepaliMode) {
      setState(() {
        _isNepaliMode = true;
        _selectedLanguage = 'ne';
      });
    } else if (englishCount > 0 && nepaliCount == 0 && _isNepaliMode) {
      setState(() {
        _isNepaliMode = false;
        _selectedLanguage = 'en';
      });
    }
  }

  // Helper method to count matches of a pattern in a string
  int _countMatches(String text, RegExp pattern) {
    return pattern.allMatches(text).length;
  }

  // Method to validate language consistency
  bool _validateLanguageConsistency() {
    final text = contentController.text;
    if (text.isEmpty) return true;

    bool hasNepaliChars = _nepaliRange.hasMatch(text);
    bool hasEnglishChars = _englishRange.hasMatch(text);

    // Check if text contains only numeric values
    bool isOnlyNumeric = RegExp(r'^[0-9\s]+$').hasMatch(text.trim());
    if (isOnlyNumeric) {
      ToastHelper.showError(
        context,
        'Journal content cannot contain only numbers. Please add some text.',
      );
      return false;
    }

    // If the content has Nepali characters but language is set to English
    if (hasNepaliChars && !_isNepaliMode) {
      ToastHelper.showError(
        context,
        'Text contains Nepali characters. Please switch to Nepali mode.',
      );
      return false;
    }

    // If the content has English characters but language is set to Nepali
    if (hasEnglishChars && _isNepaliMode) {
      ToastHelper.showError(
        context,
        'Text contains English characters. Please switch to English mode.',
      );
      return false;
    }

    return true;
  }

  @override
  Widget build(BuildContext context) {
    final cardColor = widget.isDarkMode ? Colors.grey[800] : Colors.white;
    final textColor = widget.isDarkMode ? Colors.white : Colors.black;
    final labelColor = widget.isDarkMode ? Colors.white70 : Colors.blueGrey;
    final borderColor = widget.isDarkMode ? Colors.white24 : Colors.blueGrey;

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Header Row with fixed height
          Container(
            height: 48,
            margin: EdgeInsets.only(bottom: 8),
            child: Row(
              children: [
                Icon(
                  editingJournalId == null ? Icons.create : Icons.edit,
                  size: 20,
                  color: labelColor,
                ),
                SizedBox(width: 8),
                Text(
                  editingJournalId == null
                      ? 'Create New Journal'
                      : 'Edit Journal',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: labelColor,
                  ),
                ),
                Spacer(),
                // Language toggle
                GestureDetector(
                  onTap: _toggleLanguage,
                  child: Container(
                    height: 36,
                    padding: EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(18),
                      color: (_isNepaliMode ? Colors.green : Colors.blue)
                          .withOpacity(0.1),
                      border: Border.all(
                        color: _isNepaliMode ? Colors.green : Colors.blue,
                        width: 1,
                      ),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.language,
                          size: 16,
                          color: _isNepaliMode ? Colors.green : Colors.blue,
                        ),
                        SizedBox(width: 4),
                        Text(
                          _isNepaliMode ? 'नेपाली' : 'English',
                          style: TextStyle(
                            fontSize: 13,
                            color: _isNepaliMode ? Colors.green : Colors.blue,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                SizedBox(width: 8),
                // Microphone button
                Container(
                  height: 36,
                  width: 36,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: _isListening
                        ? Colors.red.withOpacity(0.1)
                        : Colors.grey.withOpacity(0.1),
                  ),
                  child: IconButton(
                    padding: EdgeInsets.zero,
                    icon: Icon(
                      _isListening ? Icons.mic : Icons.mic_none,
                      color: _isListening ? Colors.red : labelColor,
                      size: 20,
                    ),
                    onPressed: _isListening ? _stopListening : _startListening,
                    tooltip:
                        _isListening ? 'Stop Recording' : 'Start Recording',
                  ),
                ),
              ],
            ),
          ),
          if (_isListening)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(
                'Listening... Speak in ${_isNepaliMode ? 'Nepali' : 'English'}',
                style: TextStyle(
                  color: Colors.red,
                  fontSize: 13,
                  fontStyle: FontStyle.italic,
                ),
              ),
            ),
          // Form Card
          Card(
            elevation: 2,
            color: cardColor,
            margin: EdgeInsets.zero,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
              side: BorderSide(color: borderColor.withOpacity(0.1)),
            ),
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Title Field
                  TextField(
                    controller: titleController,
                    style: TextStyle(color: textColor, fontSize: 15),
                    decoration: InputDecoration(
                      labelText: _isNepaliMode ? 'शीर्षक' : 'Title',
                      labelStyle: TextStyle(color: labelColor, fontSize: 14),
                      contentPadding: EdgeInsets.all(12),
                      enabledBorder: OutlineInputBorder(
                        borderSide: BorderSide(color: borderColor),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderSide: BorderSide(
                          color: _isNepaliMode ? Colors.green : Colors.blue,
                          width: 2,
                        ),
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                  SizedBox(height: 12),
                  // Content Field
                  Container(
                    height: 200, // Fixed height for content
                    child: TextField(
                      controller: contentController,
                      style: TextStyle(color: textColor, fontSize: 15),
                      maxLines: null,
                      expands: true,
                      textAlignVertical: TextAlignVertical.top,
                      decoration: InputDecoration(
                        labelText: _isNepaliMode ? 'सामग्री' : 'Content',
                        alignLabelWithHint: true,
                        labelStyle: TextStyle(color: labelColor, fontSize: 14),
                        contentPadding: EdgeInsets.all(12),
                        enabledBorder: OutlineInputBorder(
                          borderSide: BorderSide(color: borderColor),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        focusedBorder: OutlineInputBorder(
                          borderSide: BorderSide(
                            color: _isNepaliMode ? Colors.green : Colors.blue,
                            width: 2,
                          ),
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          SizedBox(height: 12),
          // Action Buttons
          if (editingJournalId != null)
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _clearForm,
                    icon: Icon(Icons.cancel, size: 18),
                    label: Text(
                      'Cancel',
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.grey[300],
                      foregroundColor: Colors.grey[800],
                      padding: EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                      elevation: 1,
                    ),
                  ),
                ),
                SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _handleSubmit,
                    icon: Icon(Icons.update, size: 18),
                    label: Text(
                      _isNepaliMode
                          ? 'जर्नल अपडेट गर्नुहोस्'
                          : 'Update Journal',
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue,
                      foregroundColor: Colors.white,
                      padding: EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                      elevation: 1,
                    ),
                  ),
                ),
              ],
            )
          else
            ElevatedButton.icon(
              onPressed: _handleSubmit,
              icon: Icon(Icons.save, size: 18),
              label: Text(
                _isNepaliMode ? 'जर्नल सेभ गर्नुहोस्' : 'Save Journal',
                style: TextStyle(fontSize: 15, fontWeight: FontWeight.w500),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
                padding: EdgeInsets.symmetric(vertical: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                elevation: 1,
              ),
            ),
        ],
      ),
    );
  }

  Future<void> _initializeSpeech() async {
    if (!mounted) return;
    bool available = await _speech.initialize();
    if (available && mounted) {
      setState(() {});
    }
  }

  void _startListening() async {
    if (!mounted) return;
    bool available = await _speech.initialize();
    if (available && mounted) {
      setState(() => _isListening = true);
      _speech.listen(
        onResult: (result) {
          if (!mounted) return;
          setState(() {
            _recognizedText = result.recognizedWords;
            contentController.text = _recognizedText;
          });
        },
        localeId: _isNepaliMode ? 'ne-NP' : 'en-US',
      );
    }
  }

  void _stopListening() {
    if (!mounted) return;
    _speech.stop();
    setState(() => _isListening = false);
  }

  void _toggleLanguage() {
    if (!mounted) return;
    setState(() {
      _isNepaliMode = !_isNepaliMode;
      _selectedLanguage = _isNepaliMode ? 'ne' : 'en';
    });
  }

  @override
  void dispose() {
    titleController.dispose();
    contentController.dispose();
    flutterTts.stop();
    super.dispose();
  }

  void _clearForm() {
    if (!mounted) return;
    setState(() {
      editingJournalId = null;
      titleController.clear();
      contentController.clear();
      _recognizedText = '';
    });
  }

  void _handleSubmit() {
    // Check if fields are empty
    if (titleController.text.trim().isEmpty ||
        contentController.text.trim().isEmpty) {
      ToastHelper.showError(context, 'Please fill in all fields');
      return;
    }

    // Check if title contains only numeric characters
    if (RegExp(r'^[0-9\s]+$').hasMatch(titleController.text.trim())) {
      ToastHelper.showError(
        context,
        'Title cannot contain only numbers. Please add some text.',
      );
      return;
    }

    // Validate language consistency before submission
    if (!_validateLanguageConsistency()) {
      return;
    }

    final journal = {
      'title': titleController.text.trim(),
      'content': contentController.text.trim(),
      'language': _selectedLanguage,
    };

    try {
      if (editingJournalId == null) {
        widget.onSubmit(journal);
      } else {
        widget.onUpdate(journal, editingJournalId!);
      }
      _clearForm(); // Clear form after successful submission
    } catch (e) {
      ToastHelper.showError(context, 'Failed to save journal: $e');
    }
  }

  void setEditingJournal(Map<String, dynamic> journal) {
    if (!mounted) return;
    setState(() {
      editingJournalId = journal['id'].toString();
      titleController.text = journal['title'] ?? '';
      contentController.text = journal['content'] ?? '';
      _isNepaliMode = journal['language'] == 'ne';
      _selectedLanguage = _isNepaliMode ? 'ne' : 'en';
    });
  }

  Map<String, dynamic> getJournalData() {
    return {
      'title': titleController.text,
      'content': contentController.text,
      'language': _selectedLanguage,
    };
  }

  Future<void> _speakText() async {
    await flutterTts.setLanguage(_isNepaliMode ? 'ne-NP' : 'en-US');
    await flutterTts.setPitch(1.0);
    await flutterTts.speak(contentController.text);
  }
}
