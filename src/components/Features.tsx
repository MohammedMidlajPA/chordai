import { Brain, Zap, Music, Download, Sparkles, ArrowRight } from "lucide-react";

const Features = () => {
  const features = [
    {
      icon: Brain,
      title: "AI-Powered Analysis",
      description: "Advanced algorithms analyze audio patterns to detect chords from your guitar recordings and audio files.",
      color: "text-primary",
      bgColor: "bg-primary/10"
    },
    {
      icon: Zap,
      title: "Fast Processing",
      description: "Get chord progressions quickly after uploading. Our system processes audio efficiently in your browser.",
      color: "text-accent",
      bgColor: "bg-accent/10"
    },
    {
      icon: Music,
      title: "Multiple Formats",
      description: "Support for MP3, WAV, FLAC, and more. Works with audio files up to 50MB in size.",
      color: "text-primary",
      bgColor: "bg-primary/10"
    },
    {
      icon: Download,
      title: "Guitar Chord Diagrams",
      description: "View detected chords as guitar fingering diagrams. Perfect for learning and practicing songs.",
      color: "text-accent",
      bgColor: "bg-accent/10"
    }
  ];

  return (
    <section id="features" className="py-32 bg-background relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-0 w-72 h-72 rounded-full bg-primary/5 blur-3xl" />
        <div className="absolute bottom-1/4 right-0 w-96 h-96 rounded-full bg-accent/5 blur-3xl" />
      </div>

      <div className="container mx-auto px-4 relative z-10">
        <div className="text-center mb-20 animate-slide-up opacity-0">
          <div className="inline-flex items-center space-x-2 premium-card px-4 py-2 mb-6 text-sm">
            <Sparkles className="h-4 w-4 text-primary" />
            <span className="text-foreground font-medium">Powerful Features</span>
          </div>
          
          <h2 className="font-sora font-bold text-4xl md:text-6xl mb-6 text-foreground leading-tight">
            Everything You Need for
            <span className="block text-transparent bg-gradient-to-r from-primary to-accent bg-clip-text">
              Guitar Chord Detection
            </span>
          </h2>
          <p className="text-muted-foreground text-xl max-w-3xl mx-auto leading-relaxed">
            Tools designed to help guitarists analyze and learn chord progressions from any audio recording.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <div 
              key={index}
              className="premium-card p-8 group hover:scale-[1.02] transition-all duration-500 text-center animate-slide-up opacity-0"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className={`inline-flex items-center justify-center w-20 h-20 rounded-3xl ${feature.bgColor} mb-6 group-hover:scale-110 transition-all duration-300`}>
                <feature.icon className={`h-10 w-10 ${feature.color}`} />
              </div>
              <h3 className="font-sora font-semibold text-xl mb-4 text-foreground group-hover:text-primary transition-colors">
                {feature.title}
              </h3>
              <p className="text-muted-foreground leading-relaxed mb-4">
                {feature.description}
              </p>
              <ArrowRight className="h-4 w-4 text-primary mx-auto opacity-0 group-hover:opacity-100 transition-all duration-300 group-hover:translate-x-1" />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
