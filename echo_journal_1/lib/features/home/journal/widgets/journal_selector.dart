import 'package:flutter/material.dart';

class LanguageSelectorWidget extends StatefulWidget {
  final VoidCallback onRefresh;
  final String selectedLanguage;
  final Function(String) onLanguageChanged;

  const LanguageSelectorWidget({
    super.key,
    required this.onRefresh,
    required this.selectedLanguage,
    required this.onLanguageChanged,
  });

  @override
  State<LanguageSelectorWidget> createState() => _LanguageSelectorWidgetState();
}

class _LanguageSelectorWidgetState extends State<LanguageSelectorWidget> {
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(top: 0, bottom: 2.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              Text(
                'Language:',
                style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
              ),
              SizedBox(width: 8),
              Container(
                height: 26,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(8),
                  color: Colors.white,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black12,
                      blurRadius: 4,
                      offset: Offset(0, 2),
                    ),
                  ],
                ),
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<String>(
                    value: widget.selectedLanguage,
                    isDense: true,
                    padding: EdgeInsets.zero,
                    items: [
                      DropdownMenuItem(
                        value: 'en',
                        child: Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 8.0),
                          child: Text(
                            'English',
                            style: TextStyle(fontSize: 13),
                          ),
                        ),
                      ),
                      DropdownMenuItem(
                        value: 'ne',
                        child: Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 8.0),
                          child: Text('Nepali', style: TextStyle(fontSize: 13)),
                        ),
                      ),
                    ],
                    onChanged: (String? newValue) {
                      if (newValue != null) {
                        widget.onLanguageChanged(newValue);
                      }
                    },
                  ),
                ),
              ),
            ],
          ),
          IconButton(
            icon: Icon(Icons.refresh, color: Colors.blue, size: 18),
            onPressed: widget.onRefresh,
            tooltip: 'Refresh Journals',
            padding: EdgeInsets.zero,
            constraints: BoxConstraints(minWidth: 28, minHeight: 28),
          ),
        ],
      ),
    );
  }
}
