"""
Statistics Panel Component for displaying detection results.
"""

from typing import Dict, Any, Optional
import customtkinter as ctk


class StatsPanel(ctk.CTkFrame):
    """
    Panel for displaying detection statistics.
    """
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the stats panel UI."""
        # Configure grid
        self.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text="📊 Statistics",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        self.title_label.grid(row=0, column=0, padx=10, pady=(5, 0), sticky="w", columnspan=5)
        
        # Stats labels
        self.team0_label = ctk.CTkLabel(
            self,
            text="Team 0: -",
            font=ctk.CTkFont(size=12)
        )
        self.team0_label.grid(row=1, column=0, padx=10, pady=10)
        
        self.team1_label = ctk.CTkLabel(
            self,
            text="Team 1: -",
            font=ctk.CTkFont(size=12)
        )
        self.team1_label.grid(row=1, column=1, padx=10, pady=10)
        
        self.unassigned_label = ctk.CTkLabel(
            self,
            text="Unassigned: -",
            font=ctk.CTkFont(size=12)
        )
        self.unassigned_label.grid(row=1, column=2, padx=10, pady=10)
        
        self.referee_label = ctk.CTkLabel(
            self,
            text="Referee: -",
            font=ctk.CTkFont(size=12)
        )
        self.referee_label.grid(row=1, column=3, padx=10, pady=10)
        
        self.ball_label = ctk.CTkLabel(
            self,
            text="Ball: -",
            font=ctk.CTkFont(size=12)
        )
        self.ball_label.grid(row=1, column=4, padx=10, pady=10)
    
    def update_stats(self, stats: Optional[Dict[str, Any]]):
        """Update displayed statistics."""
        if stats is None:
            self._clear_stats()
            return
        
        # Update team counts
        team0 = stats.get("team_0", 0)
        team1 = stats.get("team_1", 0)
        unassigned = stats.get("unassigned", 0)
        
        self.team0_label.configure(text=f"Team 0: {team0}")
        self.team1_label.configure(text=f"Team 1: {team1}")
        self.unassigned_label.configure(text=f"Unassigned: {unassigned}")
        
        # Get class breakdown
        classes = stats.get("classes", {})
        referee_count = classes.get("Referee", 0)
        ball_count = classes.get("Ball", 0)
        
        self.referee_label.configure(text=f"Referee: {referee_count}")
        self.ball_label.configure(text=f"Ball: {ball_count}")
    
    def _clear_stats(self):
        """Clear all statistics."""
        self.team0_label.configure(text="Team 0: -")
        self.team1_label.configure(text="Team 1: -")
        self.unassigned_label.configure(text="Unassigned: -")
        self.referee_label.configure(text="Referee: -")
        self.ball_label.configure(text="Ball: -")
