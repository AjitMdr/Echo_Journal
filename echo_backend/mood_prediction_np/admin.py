from django.contrib import admin
from .models import SentimentAnalysis, BERTModel

@admin.register(SentimentAnalysis)
class SentimentAnalysisAdmin(admin.ModelAdmin):
    list_display = ('id', 'text_preview', 'sentiment_display', 'confidence', 'created_at')
    list_filter = ('sentiment', 'created_at')
    search_fields = ('text',)
    readonly_fields = ('created_at',)
    
    def text_preview(self, obj):
        if len(obj.text) > 50:
            return f"{obj.text[:50]}..."
        return obj.text
    text_preview.short_description = 'Text'
    
    def sentiment_display(self, obj):
        sentiment_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        return sentiment_map.get(obj.sentiment, "Unknown")
    sentiment_display.short_description = 'Sentiment'

@admin.register(BERTModel)
class BERTModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'accuracy', 'f1_score', 'training_completed', 'is_active')
    list_filter = ('is_active', 'training_completed')
    search_fields = ('name', 'description')
    readonly_fields = ('accuracy', 'f1_score', 'training_completed')
    
    actions = ['make_active']
    
    def make_active(self, request, queryset):
        # Only make one model active
        if queryset.count() > 1:
            self.message_user(request, "Please select only one model to make active.", level='error')
            return
        
        # Deactivate all models
        BERTModel.objects.filter(is_active=True).update(is_active=False)
        
        # Activate the selected model
        model = queryset.first()
        model.is_active = True
        model.save()
        
        self.message_user(request, f"Model '{model.name}' is now active.")
    make_active.short_description = "Mark selected model as active"