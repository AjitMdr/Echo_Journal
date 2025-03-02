class Journal {
  final int id;
  final String title;
  final String content;
  final String language;
  final DateTime createdAt;
  final DateTime updatedAt;

  Journal({
    required this.id,
    required this.title,
    required this.content,
    required this.language,
    required this.createdAt,
    required this.updatedAt,
  });

  factory Journal.fromJson(Map<String, dynamic> json) {
    return Journal(
      id: json['id'],
      title: json['title'],
      content: json['content'],
      language: json['language'],
      createdAt: DateTime.parse(json['created_at']),
      updatedAt: DateTime.parse(json['updated_at']),
    );
  }
}
